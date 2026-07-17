#!/usr/bin/env python3
"""
mp4_gs_extractor.py
===================
일반 카메라 mp4 영상 또는 Insta360 equirectangular mp4에서
COLMAP/3DGS용 이미지 추출과 사람 마스킹을 지원합니다.

사용 예:
  python mp4_gs_extractor.py --input video.mp4 --output out_dir
  python mp4_gs_extractor.py --input video.mp4 --output out_dir --interval 1.5 --no-mask
  python mp4_gs_extractor.py --input video.mp4 --output out_dir --mode normal --run-colmap
  python mp4_gs_extractor.py --input video.mp4 --output out_dir --start-from-brush  # Brush만 재시작

기능:
  - 입력 MP4가 일반 영상인지 equirectangular(2:1)인지 자동 판별
  - 일반 mp4는 시간 간격 기반으로 프레임 추출
  - 360 영상은 기존 Insta360 v360-perspective 추출
  - 추출 이미지 블러 필터 + YOLO person 마스크 처리
  - COLMAP feature_extractor 자동 실행 옵션

주의(Insta360 입력):
  - 반드시 Insta360 Studio/앱에서 "스티칭(stitch)"까지 완료한 equirectangular
    mp4(가로:세로 = 2:1)를 입력해야 함. 카메라에서 바로 나온 미가공 dual-fisheye
    원본(.insv 등)을 넣으면 자동 판별과 v360 투영이 의미 없는 결과를 만든다.

필요 패키지:
  pip install ultralytics opencv-python numpy
  ffmpeg, ffprobe, (옵션) colmap이 PATH에 있어야 함
"""

import argparse
import json
import math
import os
import subprocess
import sys
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

try:
    import cv2  # type: ignore[import]
except ImportError:
    print("[!] opencv-python 미설치. 설치: pip install opencv-python")
    cv2 = None

try:
    import numpy as np  # type: ignore[import]
except ImportError:
    print("[!] numpy 미설치. 설치: pip install numpy")
    np = None

_LOG_FILE_PATH = None


def set_log_file(path: Path) -> None:
    """이후 log()로 남기는 모든 메시지를 이 파일에 남긴다. 같은 이름의 로그가 이미
    있으면(예: 같은 출력 폴더에서 재실행) 거기에 이어 쓰지 않고, 파일명 뒤에
    실행 시각(시분초)을 붙여 매 실행마다 새 로그 파일을 만든다 - 그래야 여러 번
    돌린 로그가 한 파일에 섞여 어느 실행의 결과인지 헷갈리는 일이 없다."""
    global _LOG_FILE_PATH
    if path.exists():
        path = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
    _LOG_FILE_PATH = path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # BOM 없는 UTF-8은 메모장 등 일부 Windows 편집기가 시스템 코드페이지
        # (한글 Windows는 CP949)로 잘못 추정해 한글이 깨져 보인다. 새로 만드는
        # 파일이므로 BOM을 한 번 남겨 UTF-8임을 명시한다.
        with open(path, "wb") as f:
            f.write(b"\xef\xbb\xbf")
    except OSError:
        pass


def log(msg: str = "") -> None:
    print(msg)
    if _LOG_FILE_PATH is not None:
        try:
            with open(_LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass


SCENE_PRESETS = {
    "outdoor_wide": {
        "yaw_list": [0, 60, 120, 180, 240, 300],
        "pitch_list": [-10, 10],
        "fov": 100,
        "fps": 2,
        "out_w": 1920,
        "out_h": 1080,
    },
    "indoor_dense": {
        "yaw_list": [0, 45, 90, 135, 180, 225, 270, 315],
        "pitch_list": [-15, 0, 15],
        "fov": 90,
        "fps": 3,
        "out_w": 1920,
        "out_h": 1080,
    },
    "corridor": {
        "yaw_list": [0, 45, 90, 135, 180, 225, 270, 315],
        "pitch_list": [-20, 0],
        "fov": 80,
        "fps": 4,
        "out_w": 1920,
        "out_h": 1080,
    },
    "preview_fast": {
        "yaw_list": [0, 90, 180, 270],
        "pitch_list": [0],
        "fov": 100,
        "fps": 1,
        "out_w": 1280,
        "out_h": 720,
    },
}

PERSON_CLASS_ID = 0


def resolve_command(cmd):
    if not isinstance(cmd, (list, tuple)) or not cmd:
        return cmd

    tool = cmd[0]
    if tool in {"ffmpeg", "ffprobe", "colmap"}:
        candidates = [tool]
        if os.name == "nt":
            candidates.extend([f"{tool}.exe", f"{tool}.bat"])
        else:
            candidates.extend([f"{tool}.exe"])
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return [resolved, *cmd[1:]]
    elif tool in {"brush", "brush.exe"}:
        candidates = [tool]
        if os.name == "nt":
            candidates.extend(["brush.exe", "brush"])
        else:
            candidates.extend(["brush", "brush.exe"])
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return [resolved, *cmd[1:]]
    return list(cmd)


def resolve_executable(executable: str) -> str:
    if os.path.isabs(executable) or executable.startswith(".") or "/" in executable or "\\" in executable:
        return executable
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if os.name == "nt" and not executable.lower().endswith(".exe"):
        alt = f"{executable}.exe"
        resolved_alt = shutil.which(alt)
        if resolved_alt:
            return resolved_alt
    return executable


def run(cmd, **kwargs):
    resolved_cmd = resolve_command(cmd)
    log(f"[cmd] {' '.join(str(c) for c in resolved_cmd)}")
    try:
        return subprocess.run(resolved_cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        if e.stdout:
            log("---- stdout ----")
            log(e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="replace"))
        if e.stderr:
            log("---- stderr ----")
            log(e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace"))
        raise


def ffprobe_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration:format=duration",
        "-of", "json",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    # 가변 프레임레이트 등 일부 컨테이너는 stream에 duration이 없고 format에만 있음.
    if not float(stream.get("duration", 0.0) or 0.0):
        stream["duration"] = data.get("format", {}).get("duration", stream.get("duration", 0.0))
    return stream


def parse_resolution(resolution: str):
    if "x" not in resolution:
        raise ValueError("--resize 값은 WxH 형식이어야 합니다.")
    w, h = resolution.lower().split("x")
    return int(w), int(h)


def require_runtime_deps() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python과 numpy가 필요합니다. pip install opencv-python numpy")


def extract_sample_frame(video_path: str, out_path: Path, time_offset: float = 0.5) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(time_offset),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "format=yuvj420p",
        "-q:v", "2",
        str(out_path),
        "-loglevel", "error",
    ]
    run(cmd, capture_output=True, text=True)
    return out_path


def inspect_video_sample(video_path: str) -> dict:
    require_runtime_deps()
    tmp_dir = Path(tempfile.mkdtemp(prefix="video_sample_", dir=str(Path.cwd())))
    sample_path = tmp_dir / "sample.jpg"
    try:
        extract_sample_frame(video_path, sample_path, time_offset=0.5)
        img = cv2.imread(str(sample_path))
        if img is None:
            return {"width": 0, "height": 0, "aspect": 0.0, "is_equirectangular": False, "is_smartphone": False}
        h, w = img.shape[:2]
        aspect = float(w) / float(h)
        is_equirect = abs(aspect - 2.0) < 0.12
        is_phone = (w, h) in [(1920, 1080), (1080, 1920), (1440, 2560), (2560, 1440), (3840, 2160), (2160, 3840)]
        return {
            "width": w,
            "height": h,
            "aspect": aspect,
            "is_equirectangular": is_equirect,
            "is_smartphone": is_phone,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def is_equirectangular(stream: dict, sample_info: Optional[dict] = None) -> bool:
    width = int(stream["width"])
    height = int(stream["height"])
    aspect = float(width) / float(height)
    if sample_info:
        if sample_info.get("aspect", 0.0):
            return abs(aspect - 2.0) < 0.12 or abs(sample_info["aspect"] - 2.0) < 0.12
    return abs(aspect - 2.0) < 0.08


def normalize_yaw(yaw: float) -> float:
    return ((yaw + 180) % 360) - 180


def build_timestamps(duration: float, interval: float, start_time: float = 0.0, trim_end: float = 0.0):
    end = max(0.0, duration - trim_end)
    if start_time >= end:
        raise ValueError("start_time + trim_end >= 영상 길이입니다.")
    timestamps = []
    t = start_time
    while t <= end + 1e-6:
        timestamps.append(round(t, 3))
        t += interval
    return timestamps


def recommend_interval(duration: float) -> float:
    target = 120
    interval = duration / target
    return round(max(0.5, min(3.0, interval)), 2)


def recommend_equirect_fps(duration: float, num_views: int, target_total_frames: int, min_fps: float = 0.1, max_fps: float = 5.0) -> float:
    """normal 모드의 recommend_interval과 같은 목적: 영상 길이가 늘어나도
    (yaw/pitch 뷰 수 x fps x duration)로 정해지는 전체 프레임 수가 무한정 커지지
    않도록, 목표 총 프레임 수(target_total_frames)에 맞춰 fps를 역산한다.
    프리셋의 fps를 고정값으로 그대로 쓰면 긴 영상에서 뷰 수만큼 배로 불어나
    COLMAP이 감당하기 어려울 만큼 이미지가 쏟아진다."""
    if duration <= 0 or num_views <= 0:
        return max_fps
    fps = (target_total_frames / num_views) / duration
    return round(max(min_fps, min(max_fps, fps)), 3)


def parse_float_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def extract_normal_frames(video_path: str, out_dir: Path, timestamps, out_w: int, out_h: int, keep_aspect: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for idx, ts in enumerate(timestamps, 1):
        # 폭 10(정수부 6자리)으로 고정: 폭 7이면 1000초(약 16.7분) 이상부터 자릿수가
        # 늘어나 "999.500" > "1000.000"처럼 문자열 정렬이 실제 시간 순서와 어긋나고,
        # 이 정렬 순서에 의존하는 COLMAP sequential matcher가 깨진다.
        fname = f"T{ts:010.3f}.jpg"
        out_path = out_dir / fname
        if out_w and out_h:
            if keep_aspect:
                # 가로(out_w)를 기준으로 비율을 유지하면서 스케일하고, 세로는
                # 원본 비율에 맞춰 자동 계산한다(-2 = 짝수로 보정, 코덱 요구사항).
                # out_h는 무시되지만(원본 비율이 우선), 이렇게 해야 좌우/상하로
                # 눌리는 왜곡 없이 모든 프레임이 동일한 비율로 일관되게 나온다.
                vf = f"scale={out_w}:-2"
            else:
                # 명시적으로 비율을 무시하고 정확히 out_w x out_h로 강제 (왜곡 발생 가능)
                vf = f"scale={out_w}:{out_h}"
        else:
            vf = None
        # mjpeg가 지원하지 않는 포맷(HDR 10bit, 4:2:2 등)의 프레임이 들어오는
        # 것에 대비해 항상 mjpeg 호환 포맷으로 명시 변환한다.
        vf = f"{vf},format=yuvj420p" if vf else "format=yuvj420p"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-frames:v", "1",
        ]
        if vf:
            cmd.extend(["-vf", vf])
        cmd.extend(["-q:v", "2", str(out_path), "-loglevel", "error"])
        try:
            run(cmd, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            # 영상 끝부분 등 특정 타임스탬프에서만 디코딩/인코딩이 실패하는
            # 경우가 있다(예: 컨테이너 duration이 실제 디코드 가능한 길이보다
            # 길게 잡혀 있어 마지막 프레임 근처에서 시크가 빗나가는 경우).
            # 프레임 하나 때문에 전체 추출이 중단되지 않도록 건너뛴다.
            log(f"[!] t={ts}s 프레임 추출 실패, 건너뜁니다 ({idx}/{len(timestamps)})")
            continue
        if not out_path.exists():
            # ffmpeg는 위와 같은 시크 실패 상황에서도 종료 코드 0으로 끝나면서
            # 프레임 파일을 전혀 만들지 않는 경우가 있다. 이때 out_path를 그대로
            # extracted에 넣으면 이후 단계(rename/imread 등)가 존재하지 않는
            # 파일을 참조하다 터진다.
            log(f"[!] t={ts}s 프레임 파일이 생성되지 않음, 건너뜁니다 ({idx}/{len(timestamps)})")
            continue
        extracted.append(out_path)
    return extracted


def compute_square_pixel_fov(h_fov_deg: float, out_w: int, out_h: int) -> float:
    """
    h_fov(가로 시야각)를 기준으로, 출력 비율(out_h/out_w)에 맞는 v_fov를 계산.
    f_x = (out_w/2) / tan(h_fov/2) 와 f_y = (out_h/2) / tan(v_fov/2) 가 같아지도록
    (= 정사각형 픽셀, 왜곡 없음) v_fov를 역산한다.

    h_fov=v_fov 로 같은 각도를 주고 out_w != out_h 인 경우, 가로/세로 픽셀 밀도가
    달라져서 "좌우로 눌린" 형태로 찌그러지는 문제가 생긴다. 이 함수로 계산한
    v_fov를 쓰면 f_x == f_y가 보장되어 왜곡이 없어진다.
    """
    half_h_fov_rad = math.radians(h_fov_deg / 2.0)
    fx = (out_w / 2.0) / math.tan(half_h_fov_rad)
    half_v_fov_rad = math.atan((out_h / 2.0) / fx)
    return math.degrees(half_v_fov_rad) * 2.0


def compute_pinhole_intrinsics(h_fov_deg: float, out_w: int, out_h: int):
    """compute_square_pixel_fov와 동일한 가정(fx==fy, 정사각형 픽셀)으로 COLMAP
    PINHOLE camera_params(fx,fy,cx,cy)를 만든다. v360 필터로 뽑은 뷰는 h_fov/out_w/out_h가
    이미 정확히 정해져 있으므로, COLMAP이 초점거리를 별도로 추정하게 둘 필요가 없다."""
    half_h_fov_rad = math.radians(h_fov_deg / 2.0)
    fx = (out_w / 2.0) / math.tan(half_h_fov_rad)
    return fx, fx, out_w / 2.0, out_h / 2.0


def extract_view(video_path: str, out_dir: Path, yaw: float, pitch: float, fov: float, fps: int, out_w: int, out_h: int, tag: str, start_time: float = 0.0, clip_duration: float = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{tag}_%05d.jpg")
    yaw_norm = normalize_yaw(yaw)

    # h_fov는 프리셋 값을 그대로 사용(가로 시야각 기준), v_fov는 출력 비율에 맞춰
    # 자동 계산해서 정사각형 픽셀을 보장한다 (anamorphic 왜곡 방지).
    h_fov = fov
    v_fov = compute_square_pixel_fov(h_fov, out_w, out_h)

    vf = (
        f"v360=e:flat:yaw={yaw_norm}:pitch={pitch}:roll=0:"
        f"h_fov={h_fov}:v_fov={v_fov:.4f}:w={out_w}:h={out_h},"
        f"fps={fps},format=yuvj420p"
    )
    cmd = ["ffmpeg", "-y"]
    if start_time > 0:
        # -i 앞에 두는 input seek(-ss)은 fast seek라 프레임 단위로 정확하진 않지만,
        # 여러 프레임을 뽑는 이 파이프라인 특성상 속도가 더 중요하다.
        cmd += ["-ss", str(start_time)]
    cmd += ["-i", video_path, "-vf", vf]
    if clip_duration is not None:
        cmd += ["-t", str(clip_duration)]
    cmd += ["-q:v", "2", pattern]
    run(cmd, capture_output=True, text=True)
    return sorted(out_dir.glob(f"{tag}_*.jpg"))


def extract_views_batch(video_path: str, out_dir: Path, views, fov: float, fps: int, out_w: int, out_h: int, start_time: float = 0.0, clip_duration: float = None):
    """같은 프리셋(fov/fps/out_w/out_h가 동일한) 여러 yaw/pitch 뷰를 한 번의 ffmpeg
    실행으로 뽑는다. split 필터로 디코딩은 한 번만 하고, 뷰마다 별도 v360 분기 +
    출력 파일로 매핑한다. yaw/pitch 뷰 수만큼 영상 전체를 반복 디코딩하는 것보다
    훨씬 빠르다 (예: indoor_dense 8x3=24뷰 -> 디코딩 24회 대신 1회).
    views: [(yaw, pitch, tag), ...]
    start_time/clip_duration: --start-time/--trim-end 적용을 위한 구간 자르기(초).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h_fov = fov
    v_fov = compute_square_pixel_fov(h_fov, out_w, out_h)

    n = len(views)
    filter_parts = [f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n))]
    for i, (yaw, pitch, _tag) in enumerate(views):
        yaw_norm = normalize_yaw(yaw)
        filter_parts.append(
            f"[s{i}]v360=e:flat:yaw={yaw_norm}:pitch={pitch}:roll=0:"
            f"h_fov={h_fov}:v_fov={v_fov:.4f}:w={out_w}:h={out_h},"
            f"fps={fps},format=yuvj420p[o{i}]"
        )
    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"]
    if start_time > 0:
        cmd += ["-ss", str(start_time)]
    cmd += ["-i", video_path, "-filter_complex", filter_complex]
    for i, (_yaw, _pitch, tag) in enumerate(views):
        pattern = str(out_dir / f"{tag}_%05d.jpg")
        cmd += ["-map", f"[o{i}]"]
        if clip_duration is not None:
            # -t는 output별 옵션이라 -map 뒤/각 출력 파일 앞에 매번 넣어야 한다.
            # -i 뒤에 한 번만 넣으면 여러 -map 출력 중 첫 번째에만 적용되고 나머지는
            # 트림 없이 끝까지 추출되는 버그가 생긴다.
            cmd += ["-t", str(clip_duration)]
        cmd += ["-q:v", "2", pattern]
    cmd += ["-loglevel", "error"]

    run(cmd, capture_output=True, text=True)
    return {tag: sorted(out_dir.glob(f"{tag}_*.jpg")) for _yaw, _pitch, tag in views}


def laplacian_variance(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_yolo_seg(model_name: str = "yolov8n-seg.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        log("[!] ultralytics 미설치. 설치: pip install ultralytics")
        sys.exit(1)
    return YOLO(model_name)


def make_person_mask(yolo_model, img, conf=0.3):
    h, w = img.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    results = yolo_model.predict(img, classes=[PERSON_CLASS_ID], conf=conf, verbose=False)
    person_pixels = 0
    for r in results:
        if getattr(r, 'masks', None) is None:
            continue
        for seg in r.masks.data.cpu().numpy():
            seg_resized = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
            person_area = seg_resized > 0.5
            mask[person_area] = 0
            person_pixels += int(person_area.sum())
    if person_pixels > 0:
        kernel = np.ones((9, 9), np.uint8)
        inv = cv2.bitwise_not(mask)
        inv = cv2.dilate(inv, kernel, iterations=1)
        mask = cv2.bitwise_not(inv)
    ratio = float((mask == 0).sum()) / (h * w)
    return mask, ratio


def report_blur_stats(raw_frames):
    variances = []
    for fp in raw_frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        variances.append(laplacian_variance(img))
    if not variances:
        log("[!] 읽을 수 있는 프레임이 없습니다.")
        return []
    arr = np.array(variances)
    log(f"[i] 블러 분산값 통계 (n={len(arr)}): min={arr.min():.1f}  p10={np.percentile(arr,10):.1f}  median={np.median(arr):.1f}  p90={np.percentile(arr,90):.1f}  max={arr.max():.1f}")
    return variances


def process_frames(raw_frames, images_dir: Path, masks_dir: Path, yolo_model, blur_thresh: float, person_skip_ratio: float, person_conf: float):
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    kept, skipped_blur, skipped_person = 0, 0, 0
    metadata_frames = []
    for fp in raw_frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        var = laplacian_variance(img)
        if var < blur_thresh:
            skipped_blur += 1
            fp.unlink(missing_ok=True)
            continue
        mask, person_ratio = make_person_mask(yolo_model, img, conf=person_conf)
        if person_ratio > person_skip_ratio:
            skipped_person += 1
            fp.unlink(missing_ok=True)
            continue
        dst_img = images_dir / fp.name
        shutil.move(str(fp), str(dst_img))
        mask_path = masks_dir / f"{fp.name}.png"
        cv2.imwrite(str(mask_path), mask)
        metadata_frames.append({
            "file": dst_img.name,
            "mask": mask_path.name,
            "blur_var": round(var, 2),
            "person_ratio": round(person_ratio, 4),
        })
        kept += 1
    return kept, skipped_blur, skipped_person, metadata_frames


def find_image_source_dir(images_dir: Path) -> Path:
    candidates = [images_dir]
    if images_dir.exists():
        candidates.extend([
            images_dir / "images",
            images_dir.parent / "images" if images_dir.parent != images_dir else None,
        ])
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_dir():
            image_files = [p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
            if image_files:
                return candidate
    return images_dir


def prepare_colmap_inputs(images_dir: Path, masks_dir: Path, colmap_images_dir: Path, colmap_masks_dir: Path):
    colmap_images_dir.mkdir(parents=True, exist_ok=True)
    colmap_masks_dir.mkdir(parents=True, exist_ok=True)
    source_dir = find_image_source_dir(images_dir)
    image_paths = sorted([p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}], key=lambda p: p.name.lower())
    if not image_paths:
        raise FileNotFoundError(f"COLMAP 입력 이미지가 없습니다: {source_dir}")
    if source_dir != images_dir:
        log(f"[i] 이미지 소스 디렉토리 자동 감지: {source_dir}")

    index_map = {}
    for idx, img_path in enumerate(image_paths, start=1):
        ext = img_path.suffix.lower()
        dst_img = colmap_images_dir / f"{idx:08d}{ext}"
        shutil.copy2(img_path, dst_img)
        index_map[dst_img.name] = img_path.name

        if masks_dir.exists():
            mask_path = masks_dir / f"{img_path.name}.png"
            if mask_path.exists():
                dst_mask = colmap_masks_dir / f"{idx:08d}{ext}.png"
                shutil.copy2(mask_path, dst_mask)

    index_map_path = colmap_images_dir.parent / "image_index.json"
    with open(index_map_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)
    return image_paths


def count_registered_images(model_dir: Path) -> int:
    result = subprocess.run(
        ["colmap", "model_analyzer", "--path", str(model_dir)],
        capture_output=True, text=True,
    )
    for line in (result.stdout + result.stderr).splitlines():
        if "Registered images:" in line:
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                return 0
    return 0


def _log_registration_ratio(label: str, count: int, total_images: int) -> None:
    if total_images:
        log(f"[i] {label}: {count}장 / 전체 {total_images}장 등록 ({count / total_images * 100:.1f}%)")


def merge_sparse_models(sparse_dir: Path, total_images: int = None) -> Path:
    """colmap mapper는 씬이 완전히 연결되지 않으면 여러 개의 분리된 서브모델
    (sparse/0, sparse/1, ...)을 만든다. model_converter/Brush는 관례적으로
    sparse/0만 사용하는데, 그대로 두면 더 큰 재구성을 조용히 버리고 훨씬 빈약한
    파편(0번)만 쓰게 될 수 있다. 등록 이미지 수가 가장 많은 모델을 기준으로,
    나머지 모델들을 colmap model_merger로 하나씩 흡수 시도한다(공통으로 등록된
    이미지가 있어야 두 모델의 좌표계를 정합할 수 있음 - 공통 이미지가 없으면
    병합이 실패하는데, 이 경우 해당 모델은 버리지 않고 별도 번호로 보존한다).
    최종적으로 가장 크게 합쳐진 모델을 sparse/0 자리에 둔다.
    total_images가 주어지면 최종적으로 전체 입력 이미지 중 몇 %가 등록됐는지도 남긴다."""
    model_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    if not model_dirs:
        raise FileNotFoundError(f"COLMAP sparse 모델이 없습니다: {sparse_dir}")

    counts = {d: count_registered_images(d) for d in model_dirs}
    for d in model_dirs:
        log(f"    sparse 모델 '{d.name}': 등록 이미지 {counts[d]}장")

    ordered = sorted(model_dirs, key=lambda d: counts[d], reverse=True)
    zero_dir = sparse_dir / "0"
    if len(ordered) == 1:
        _log_registration_ratio("최종 등록", counts[ordered[0]], total_images)
        return ordered[0]

    work_dir = sparse_dir / "_merge_work"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    current = work_dir / "current"
    shutil.copytree(ordered[0], current)
    current_count = counts[ordered[0]]
    leftover_srcs = []
    for i, other in enumerate(ordered[1:]):
        merge_out = work_dir / f"attempt_{i}"
        merge_out.mkdir(parents=True, exist_ok=True)  # model_merger는 출력 폴더를 자동 생성하지 않음
        result = subprocess.run(
            ["colmap", "model_merger",
             "--input_path1", str(current), "--input_path2", str(other),
             "--output_path", str(merge_out)],
            capture_output=True, text=True,
        )
        # model_merger는 두 모델 간 공통 등록 이미지가 부족해 정합(alignment)에
        # 실패해도 exit code 0을 반환하고 출력 폴더에 뭔가를 남길 수 있다. 그래서
        # returncode/출력 존재 여부만으로는 실제 병합 성공 여부를 판단할 수 없다.
        # colmap이 남기는 원본 진단 메시지를 그대로 기록해서 원인을 확인할 수 있게 한다.
        log(f"---- colmap model_merger 출력 ('{other.name}' 흡수 시도) ----")
        if result.stdout.strip():
            log(result.stdout.rstrip())
        if result.stderr.strip():
            log(result.stderr.rstrip())

        merge_ran = result.returncode == 0 and merge_out.exists() and any(merge_out.iterdir())
        merged_count = count_registered_images(merge_out) if merge_ran else 0

        # 진짜 병합이 됐다면 등록 이미지 수는 절대 줄어들 수 없다(정합 실패 시
        # model_merger가 더 작은 결과를 낼 수 있음). 누적 장수보다 줄었다면
        # 정합 실패로 간주해서 기존에 쌓아온 모델을 잃지 않도록 한다.
        if merge_ran and merged_count >= current_count:
            log(f"[i] sparse 모델 병합: '{other.name}'({counts[other]}장) 흡수 -> 누적 {merged_count}장")
            shutil.rmtree(current)
            current = merge_out
            current_count = merged_count
        else:
            # 공통 등록 이미지가 없으면 model_merger가 정합을 못 해 실패한다.
            # 데이터를 버리지 않고 따로 보관해서 나중에 수동으로 참고할 수 있게 한다.
            log(f"[!] sparse 모델 '{other.name}'({counts[other]}장) 병합 실패"
                f"(결과 {merged_count}장 < 누적 {current_count}장) - 공통 등록 이미지 부족으로 추정, 별도 보관")
            shutil.rmtree(merge_out, ignore_errors=True)
            leftover_copy = work_dir / f"unmerged_src_{i}"
            shutil.copytree(other, leftover_copy)
            leftover_srcs.append(leftover_copy)

    for d in model_dirs:
        shutil.rmtree(d, ignore_errors=True)

    shutil.copytree(current, zero_dir)
    for i, src in enumerate(leftover_srcs, start=1):
        shutil.copytree(src, sparse_dir / str(i))

    shutil.rmtree(work_dir, ignore_errors=True)
    final_count = count_registered_images(zero_dir)
    log(f"[i] 최종 병합 모델: {final_count}장 등록 (sparse/0)")
    _log_registration_ratio("최종 등록", final_count, total_images)
    return zero_dir


def run_colmap_pipeline(out_dir: Path, images_dir: Path, masks_dir: Path, db_path: str, matcher: str = "sequential", prepare_brush: bool = False, vocab_tree_path: str = None, camera_model: str = None, camera_params: str = None):
    colmap_dir = out_dir / "colmap"
    colmap_images_dir = colmap_dir / "images"
    colmap_masks_dir = colmap_dir / "masks"
    sparse_dir = colmap_dir / "sparse"
    export_dir = colmap_dir / "export"
    brush_dir = out_dir / "brush_prepared"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    prepare_colmap_inputs(images_dir, masks_dir, colmap_images_dir, colmap_masks_dir)
    log(f"[i] COLMAP 입력 이미지 정렬 완료: {colmap_images_dir}")

    feature_cmd = [
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", str(colmap_images_dir),
        # 한 잡의 모든 프레임은 같은 영상/같은 렌즈 설정에서 나온다. EXIF가 없는
        # ffmpeg 출력 프레임을 매 이미지마다 별도 카메라로 추정하게 두면(기본값) BA가
        # 불필요하게 불안정해지므로 카메라를 하나로 공유시킨다.
        "--ImageReader.single_camera", "1",
    ]
    if camera_model and camera_params:
        # equirect 추출은 h_fov/out_w/out_h로 fx=fy(정사각형 픽셀)를 이미 정확히 알고
        # 있으므로, COLMAP이 초점거리를 추정하게 두지 않고 그대로 넘긴다.
        feature_cmd += ["--ImageReader.camera_model", camera_model, "--ImageReader.camera_params", camera_params]
    if masks_dir.exists() and any(colmap_masks_dir.iterdir()):
        feature_cmd += ["--ImageReader.mask_path", str(colmap_masks_dir)]
    run(feature_cmd)

    if matcher == "vocab_tree":
        match_cmd = [
            "colmap", "vocab_tree_matcher",
            "--database_path", db_path,
        ]
        if vocab_tree_path:
            match_cmd += ["--VocabTreeMatching.vocab_tree_path", vocab_tree_path]
        # vocab_tree_path 미지정 시: 최신 COLMAP은 사전학습 트리를 자동 다운로드/캐싱함.
        # 구버전이면 "vocab_tree_path를 지정해야 한다"는 에러가 날 수 있으니
        # 그때는 --vocab-tree-path로 .bin 파일을 직접 지정해야 함.
    else:
        match_cmd = [
            "colmap",
            f"{matcher}_matcher",
            "--database_path", db_path,
        ]
    run(match_cmd)

    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", db_path,
        "--image_path", str(colmap_images_dir),
        "--output_path", str(sparse_dir),
    ]
    run(mapper_cmd)

    total_images = sum(1 for p in colmap_images_dir.iterdir() if p.is_file())
    best_model_dir = merge_sparse_models(sparse_dir, total_images=total_images)

    export_dir.mkdir(parents=True, exist_ok=True)
    converter_cmd = [
        "colmap", "model_converter",
        "--input_path", str(best_model_dir),
        "--output_path", str(export_dir),
        "--output_type", "TXT",
    ]
    run(converter_cmd)

    if prepare_brush:
        brush_images_dir = brush_dir / "images"
        brush_images_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(colmap_images_dir.glob("*"), key=lambda p: p.name.lower()):
            if img_path.is_file():
                shutil.copy2(img_path, brush_images_dir / img_path.name)

        images_txt = export_dir / "images.txt"
        cameras_txt = export_dir / "cameras.txt"
        if images_txt.exists() and cameras_txt.exists():
            cameras = {}
            with open(cameras_txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        cam_id = int(parts[0])
                        cameras[cam_id] = {"model": parts[1], "width": int(parts[2]), "height": int(parts[3]), "params": [float(x) for x in parts[4:]]}

            frames = []
            with open(images_txt, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f]
            for i in range(0, len(lines), 2):
                meta = lines[i].strip()
                if not meta or meta.startswith("#"):
                    continue
                parts = meta.split()
                if len(parts) < 10 or not parts[0].isdigit():
                    continue
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                name = parts[9]
                if camera_id not in cameras:
                    continue
                cam = cameras[camera_id]
                r00 = 1 - 2 * (qy * qy + qz * qz)
                r01 = 2 * (qx * qy - qz * qw)
                r02 = 2 * (qx * qz + qy * qw)
                r10 = 2 * (qx * qy + qz * qw)
                r11 = 1 - 2 * (qx * qx + qz * qz)
                r12 = 2 * (qy * qz - qx * qw)
                r20 = 2 * (qx * qz - qy * qw)
                r21 = 2 * (qy * qz + qx * qw)
                r22 = 1 - 2 * (qx * qx + qy * qy)
                transform = [
                    [r00, r01, r02, tx],
                    [r10, r11, r12, ty],
                    [r20, r21, r22, tz],
                    [0.0, 0.0, 0.0, 1.0],
                ]
                if cam["model"] == "PINHOLE" and cam["params"]:
                    focal = cam["params"][0]
                    camera_angle_x = 2.0 * np.arctan2(cam["width"] / 2.0, focal)
                    camera_angle_y = 2.0 * np.arctan2(cam["height"] / 2.0, focal)
                elif cam["model"] in ["OPENCV", "OPENCV_FISHEYE"] and cam["params"]:
                    focal = cam["params"][0]
                    camera_angle_x = 2.0 * np.arctan2(cam["width"] / 2.0, focal)
                    camera_angle_y = 2.0 * np.arctan2(cam["height"] / 2.0, focal)
                else:
                    camera_angle_x = 1.0
                    camera_angle_y = 1.0
                frames.append({
                    "file_path": f"images/{Path(name).name}",
                    "transform_matrix": transform,
                    "camera_angle_x": camera_angle_x,
                    "camera_angle_y": camera_angle_y,
                })

            with open(brush_dir / "transforms.json", "w", encoding="utf-8") as f:
                json.dump({"frames": frames}, f, ensure_ascii=False, indent=2)
            log(f"[i] Brush 전처리 완료: {brush_dir}")
        else:
            log("[!] COLMAP sparse 모델을 찾지 못해 Brush 전처리를 건너뜁니다.")

    return {
        "colmap_dir": str(colmap_dir),
        "sparse_dir": str(sparse_dir),
        "export_dir": str(export_dir),
        "brush_dir": str(brush_dir),
    }


def run_brush_training(dataset_dir: Path, brush_exe: str = None, extra_args=None):
    # Brush는 콜맵 스파스 포인트클라우드(sparse/*/points3D.*)가 있으면 거기서
    # 가우시안을 초기화하고, 없으면(transforms.json만 있는 nerfstudio 포맷)
    # 랜덤 초기화로 시작한다. 랜덤 초기화는 품질이 크게 떨어지므로, COLMAP
    # 결과가 있다면 반드시 콜맵 네이티브 폴더(images/ + sparse/0/)를 넘겨야 한다.
    has_colmap_points = any((dataset_dir / "sparse").glob("*/points3D.*")) if (dataset_dir / "sparse").exists() else False
    has_transforms = (dataset_dir / "transforms.json").exists()
    if not has_colmap_points and not has_transforms:
        raise FileNotFoundError(
            f"Brush 학습용 데이터셋을 찾을 수 없습니다: {dataset_dir} "
            "(images/ + sparse/0/{cameras,images,points3D}.* 또는 transforms.json 필요. --run-colmap을 먼저 실행하세요)"
        )
    if not has_colmap_points:
        log("[!] sparse/*/points3D.*가 없어 포인트클라우드 초기화 없이(랜덤 초기화) 학습합니다. "
              "COLMAP 결과 폴더(images/ + sparse/0/)를 넘기면 품질이 크게 좋아집니다.")
    brush_name = brush_exe or ("brush.exe" if os.name == "nt" else "brush")
    resolved_brush = resolve_executable(brush_name)
    cmd = [resolved_brush, str(dataset_dir)]
    if extra_args:
        cmd.extend(extra_args)
    run(cmd)


def process_single_video(video_path: str, args, raw_dir: Path, images_dir: Path, masks_dir: Path, filename_prefix: str = ""):
    """영상 한 편에 대해 모드 판별 -> 프레임 추출 -> 블러 필터 + 사람 마스킹까지 수행하고
    결과를 images_dir/masks_dir에 쌓는다. 같은 공간을 찍은 여러 영상을 하나의 COLMAP
    재구성으로 합칠 때는 영상마다 다른 filename_prefix를 줘서 images_dir/masks_dir을
    공유해도 파일명이 겹치지 않게 한다."""
    log(f"[i] 영상 정보 확인 중... ({video_path})")
    info = ffprobe_info(video_path)
    width = int(info["width"])
    height = int(info["height"])
    duration = float(info.get("duration", 0.0))
    aspect = float(width) / float(height)
    log(f"    해상도: {width}x{height}  fps: {info['r_frame_rate']}  길이: {duration:.2f}s")

    sample_info = inspect_video_sample(video_path)
    phone_note = " (스마트폰 영상 감지)" if sample_info.get("is_smartphone") else ""
    log(f"    샘플 프레임: {sample_info['width']}x{sample_info['height']} aspect={sample_info['aspect']:.2f}{phone_note}")

    fov = fps = yaw_list = pitch_list = None
    camera_model = None
    camera_params = None

    if args.mode == "auto":
        if is_equirectangular(info, sample_info):
            mode = "equirectangular"
            log("[i] 자동 판별: 360°/equirectangular 영상으로 감지되어 360 추출 파이프라인을 사용합니다.")
        else:
            mode = "normal"
            log("[i] 자동 판별: 일반 카메라 영상으로 감지되어 시간 간격 추출을 사용합니다.")
    else:
        mode = args.mode
        log(f"[i] 강제 모드: {mode}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    all_raw_frames = []
    failed_views = []

    if mode == "normal":
        interval = args.interval if args.interval else recommend_interval(duration)
        out_w, out_h = parse_resolution(args.resize)
        log(f"[i] 일반 MP4 추출: interval={interval}s start={args.start_time}s trim_end={args.trim_end}s resize={out_w}x{out_h}")
        timestamps = build_timestamps(duration, interval, args.start_time, args.trim_end)
        log(f"[i] 타임스탬프 수: {len(timestamps)}")
        all_raw_frames = extract_normal_frames(
            video_path, raw_dir, timestamps, out_w, out_h,
            keep_aspect=(args.resize_mode == "fit"),
        )
    else:
        preset = SCENE_PRESETS[args.preset]
        fov = args.fov if args.fov is not None else preset["fov"]
        yaw_list = parse_float_list(args.yaw_list) if args.yaw_list else preset["yaw_list"]
        pitch_list = parse_float_list(args.pitch_list) if args.pitch_list else preset["pitch_list"]
        out_w, out_h = preset["out_w"], preset["out_h"]
        num_views = len(yaw_list) * len(pitch_list)

        computed_v_fov = compute_square_pixel_fov(fov, out_w, out_h)
        log(f"[i] 360 추출 프리셋 '{args.preset}' 적용: yaw {len(yaw_list)} x pitch {len(pitch_list)} = {num_views}뷰")
        log(f"    h_fov={fov}° -> 자동계산 v_fov={computed_v_fov:.2f}° (출력 {out_w}x{out_h}, 정사각형 픽셀 보장)")

        if len(yaw_list) > 1:
            # yaw_list가 360°를 균등 분할한다는 가정 하의 대략적인 추정치.
            # (겹침 부족 = COLMAP이 인접 yaw끼리 매칭을 못 찾아 재구성이 끊길 위험)
            yaw_step = 360.0 / len(yaw_list)
            overlap_deg = fov - yaw_step
            status = "양호" if overlap_deg > 15 else ("중첩 적음, 주의" if overlap_deg > 0 else "중첩 없음! 매칭 끊길 위험")
            log(f"    yaw 간격 추정={yaw_step:.1f}° (fov={fov}° 기준) -> 인접 뷰 중첩 약 {overlap_deg:.1f}° ({status})")

        end_time = max(0.0, duration - args.trim_end)
        if args.start_time >= end_time:
            log(f"[!] start_time({args.start_time}s) + trim_end({args.trim_end}s) >= 영상 길이({duration:.2f}s)입니다.")
            sys.exit(1)
        clip_duration = (end_time - args.start_time) if (args.start_time > 0 or args.trim_end > 0) else None
        if clip_duration is not None:
            log(f"    구간 제한: start={args.start_time}s ~ end={end_time:.2f}s (trim_end={args.trim_end}s)")
        # fps 자동계산은 실제로 추출될 구간 길이(clip_duration) 기준이어야 한다.
        # 전체 영상 길이(duration)를 기준으로 계산하면 --start-time/--trim-end로
        # 잘라낸 짧은 구간에 맞지 않는 fps가 나와, 뷰당 프레임이 1장뿐인 등
        # 사실상 카메라 이동(병진) 없이 회전만 있는 이미지 세트가 되어 COLMAP이
        # 초기 이미지 쌍을 못 찾고 재구성에 실패한다.
        effective_duration = clip_duration if clip_duration is not None else duration

        if args.fps is not None:
            fps = args.fps
        else:
            # 프리셋 fps를 고정으로 쓰면 영상이 길어질수록 뷰 수만큼 배로 불어나
            # COLMAP이 감당 못할 만큼 프레임이 쏟아진다. 목표 총 프레임 수 기준으로
            # 자동 역산해서 영상 길이와 무관하게 총량을 비슷하게 맞춘다.
            fps = recommend_equirect_fps(effective_duration, num_views, args.target_frames)
            log(f"    fps 미지정: 목표 총 프레임 {args.target_frames}장 기준 자동 계산(추출 구간 {effective_duration:.2f}s 기준) "
                  f"-> fps={fps} (뷰당 약 {fps * effective_duration:.0f}장, 전체 약 {fps * effective_duration * num_views:.0f}장 예상)")

        fx, fy, cx, cy = compute_pinhole_intrinsics(fov, out_w, out_h)
        camera_model = "PINHOLE"
        camera_params = f"{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f}"
        log(f"    COLMAP camera_params(PINHOLE) 고정: fx=fy={fx:.2f} cx={cx:.1f} cy={cy:.1f}")
        t0 = time.time()
        views = [
            (yaw, pitch, f"y{int(round(yaw)):03d}_p{int(round(pitch)):+03d}")
            for yaw in yaw_list
            for pitch in pitch_list
        ]
        try:
            # 뷰마다 영상 전체를 새로 디코딩하지 않고, split 필터로 한 번만
            # 디코딩해서 모든 yaw/pitch를 동시에 뽑는다 (프리셋 뷰 수만큼 N배 느려지는 것 방지).
            log(f"[i] {len(views)}개 뷰를 단일 ffmpeg 패스로 추출 시도 (디코딩 1회)...")
            batch_result = extract_views_batch(
                video_path, raw_dir, views,
                fov, fps, out_w, out_h,
                start_time=args.start_time, clip_duration=clip_duration,
            )
            for _yaw, _pitch, tag in views:
                all_raw_frames.extend(batch_result.get(tag, []))
        except subprocess.CalledProcessError:
            # 단일 패스가 실패하면(예: 필터그래프 제한, 메모리 부족 등) 기존 방식대로
            # 뷰 하나씩 개별 재시도하며 추출한다. 느리지만 더 안전한 폴백.
            log("[!] 단일 패스 추출 실패, 뷰별 개별 추출로 폴백합니다.")
            for yaw in yaw_list:
                for pitch in pitch_list:
                    tag = f"y{int(round(yaw)):03d}_p{int(round(pitch)):+03d}"
                    log(f"[i] 추출 중: yaw={yaw} pitch={pitch}")
                    try:
                        frames = extract_view(
                            video_path, raw_dir, yaw, pitch,
                            fov, fps, out_w, out_h, tag,
                            start_time=args.start_time, clip_duration=clip_duration,
                        )
                        all_raw_frames.extend(frames)
                    except subprocess.CalledProcessError:
                        log(f"[!] yaw={yaw} pitch={pitch} 실패, 2초 대기 후 재시도")
                        time.sleep(2)
                        try:
                            frames = extract_view(
                                video_path, raw_dir, yaw, pitch,
                                fov, fps, out_w, out_h, tag,
                                start_time=args.start_time, clip_duration=clip_duration,
                            )
                            all_raw_frames.extend(frames)
                        except subprocess.CalledProcessError:
                            log(f"[!] 재시도도 실패: yaw={yaw} pitch={pitch} -> 스킵")
                            failed_views.append({"yaw": yaw, "pitch": pitch})
        log(f"[i] 총 원본 프레임 {len(all_raw_frames)}장 추출 완료 ({time.time() - t0:.1f}s)")

    if filename_prefix:
        # 여러 영상의 결과를 같은 images_dir/masks_dir에 합칠 때 파일명이 겹치지
        # 않도록, 블러/마스킹 처리 전에 원본 프레임 파일명 앞에 접두어를 붙인다.
        prefixed = []
        for fp in all_raw_frames:
            new_fp = fp.with_name(f"{filename_prefix}{fp.name}")
            fp.rename(new_fp)
            prefixed.append(new_fp)
        all_raw_frames = prefixed

    if args.no_mask:
        log("[i] --no-mask: 사람 마스킹 생략, 블러 필터만 적용")
        var_stats = report_blur_stats(all_raw_frames)
        images_dir.mkdir(parents=True, exist_ok=True)
        kept, skipped_blur, skipped_person = 0, 0, 0
        metadata_frames = []
        for fp in all_raw_frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            var = laplacian_variance(img)
            if var < args.blur_thresh:
                skipped_blur += 1
                fp.unlink(missing_ok=True)
                continue
            dst = images_dir / fp.name
            shutil.move(str(fp), str(dst))
            metadata_frames.append({"file": dst.name, "mask": None, "blur_var": round(var, 2)})
            kept += 1
    else:
        var_stats = report_blur_stats(all_raw_frames)
        log(f"[i] YOLO 모델 로드: {args.yolo_model}")
        yolo_model = load_yolo_seg(args.yolo_model)
        log("[i] 블러 필터 + 사람 마스킹 처리 중...")
        kept, skipped_blur, skipped_person, metadata_frames = process_frames(
            all_raw_frames, images_dir, masks_dir, yolo_model,
            args.blur_thresh, args.person_skip_ratio, args.person_conf,
        )

    if kept == 0 and var_stats:
        arr = np.array(var_stats)
        suggested = max(1.0, float(np.percentile(arr, 20)))
        log(f"[!] 블러 필터로 모든 프레임이 제거되었습니다 (blur_thresh={args.blur_thresh}).")
        log(f"    --blur-thresh {suggested:.1f} 정도로 낮춰서 다시 실행해보세요.")

    shutil.rmtree(raw_dir, ignore_errors=True)
    log(f"[i] 결과: 사용 {kept}장 / 블러 제외 {skipped_blur}장 / 사람과다 제외 {skipped_person}장")

    return {
        "video_path": str(video_path),
        "mode": mode,
        "duration": duration,
        "preset": args.preset if mode == "equirectangular" else None,
        "fov": fov,
        "fps": fps,
        "yaw_list": yaw_list,
        "pitch_list": pitch_list,
        "interval": args.interval if mode == "normal" else None,
        "camera_model": camera_model,
        "camera_params": camera_params,
        "failed_views": failed_views,
        "kept": kept,
        "skipped_blur": skipped_blur,
        "skipped_person": skipped_person,
        "total_raw": len(all_raw_frames),
        "metadata_frames": metadata_frames,
    }


def main():
    ap = argparse.ArgumentParser(description="MP4 영상에서 COLMAP/3DGS용 이미지 추출 + 사람 마스킹")
    ap.add_argument("--input", required=True, nargs="+",
                    help="입력 MP4 영상 경로. 공백으로 구분해 2개 이상 넘기면, 같은 공간을 찍은 "
                         "여러 영상의 프레임을 하나로 합쳐 COLMAP 재구성 한 번으로 처리한다 "
                         "(각 영상은 파일명 접두어(v0_, v1_, ...)로 구분되어 images/masks에 합쳐짐). "
                         "여러 영상은 --mode/--preset/--fov/--resize 등 추출 옵션이 모두 동일해야 한다.")
    ap.add_argument("--output", required=True, help="출력 디렉토리")
    ap.add_argument("--mode", choices=["auto", "normal", "equirectangular"], default="auto",
                    help="auto: 영상 샘플 프레임과 해상도 기준으로 일반/360 자동 선택, normal: 일반 MP4 추출, equirectangular: 360 추출")
    ap.add_argument("--preset", default="indoor_dense", choices=SCENE_PRESETS.keys(),
                    help="360 추출일 때 사용할 프리셋")
    ap.add_argument("--fov", type=float, default=None,
                    help="360 추출 h_fov(가로 시야각, °). 지정하지 않으면 프리셋 값 사용. "
                         "값을 키우면 같은 yaw 간격에서도 인접 뷰 중첩이 늘어난다.")
    ap.add_argument("--yaw-list", default=None,
                    help="쉼표로 구분된 yaw 각도 목록으로 프리셋 yaw_list를 덮어씀 (예: '0,90,180,270'). "
                         "개수를 줄이면 뷰 수가 줄어 속도가 빨라지지만, yaw 간격이 --fov보다 커지면 "
                         "인접 뷰끼리 겹치는 영역이 없어져 COLMAP 매칭이 끊길 수 있다.")
    ap.add_argument("--pitch-list", default=None,
                    help="쉼표로 구분된 pitch 각도 목록으로 프리셋 pitch_list를 덮어씀 (예: '-10,10')")
    ap.add_argument("--fps", type=float, default=None,
                    help="360 추출 fps 직접 지정(뷰 하나당). 지정하지 않으면 --target-frames와 "
                         "영상 길이/뷰 수 기준으로 자동 계산된다(긴 영상에서 프레임이 과도하게 "
                         "많아지는 것을 방지). 지정 시 자동 계산을 건너뛰고 그대로 사용.")
    ap.add_argument("--target-frames", type=int, default=150,
                    help="--fps 미지정 시 목표로 하는 전체(모든 yaw/pitch 뷰 합산) 원본 프레임 수. "
                         "기본 150장 안팎이면 COLMAP이 빠르고 안정적으로 재구성 가능. 장면이 "
                         "복잡하면 200~300 정도로 늘려도 됨.")
    ap.add_argument("--interval", type=float, default=None,
                    help="일반 MP4 추출 간격(초). 지정하지 않으면 권장값 사용")
    ap.add_argument("--start-time", type=float, default=0.0, help="추출 시작 시간(초)")
    ap.add_argument("--trim-end", type=float, default=0.0, help="끝에서 제외할 시간(초)")
    ap.add_argument("--resize", default="1920x1080", help="출력 프레임 크기(WxH). --resize-mode fit이면 가로(W) 기준으로만 적용되고 세로는 비율유지 자동계산됨")
    ap.add_argument("--resize-mode", choices=["fit", "stretch"], default="fit",
                    help="fit(기본값): 비율 유지하며 스케일(왜곡 없음, 권장). stretch: 비율 무시하고 W x H로 강제(왜곡 발생 가능, 기존 동작)")
    ap.add_argument("--blur-thresh", type=float, default=15.0,
                    help="라플라시안 분산 임계값 이하 프레임 제외")
    ap.add_argument("--person-skip-ratio", type=float, default=0.35,
                    help="사람 영역 비율이 이 값을 넘으면 프레임 제외")
    ap.add_argument("--person-conf", type=float, default=0.3,
                    help="YOLO person confidence threshold")
    ap.add_argument("--yolo-model", default="yolov8n-seg.pt")
    ap.add_argument("--no-mask", action="store_true", help="사람 마스킹을 생략하고 블러 필터만 적용")
    ap.add_argument("--run-colmap", action="store_true", help="추출 후 COLMAP 정렬/매칭/매핑 및 Brush 전처리까지 자동 실행")
    ap.add_argument("--start-from-colmap", action="store_true", help="이미지 추출 없이 기존 images/ 폴더를 사용해서 COLMAP부터 시작")
    ap.add_argument("--start-from-brush", action="store_true",
                    help="이미지 추출/COLMAP 없이 기존 <output>/colmap/ 폴더(images/+sparse/0/)로 Brush 학습만 재시작")
    ap.add_argument("--colmap-db", default=None, help="COLMAP database 경로 (기본: <output>/database.db)")
    ap.add_argument("--colmap-matcher", default=None, choices=["sequential", "exhaustive", "vocab_tree"],
                    help="COLMAP matcher 종류. 지정하지 않으면 모드별로 자동 선택된다 "
                         "(normal: sequential, equirectangular: exhaustive). equirectangular는 "
                         "서로 다른 yaw/pitch 뷰가 파일명 정렬 순서상 번갈아 오므로 sequential을 쓰면 "
                         "실제로 겹치는 뷰끼리 매칭되지 않아 재구성이 조각날 수 있다.")
    ap.add_argument("--vocab-tree-path", default=None,
                    help="vocab_tree matcher용 사전학습 vocabulary tree(.bin) 경로. "
                         "지정 안 하면 최신 COLMAP은 자동 다운로드/캐싱을 시도함 "
                         "(구버전 COLMAP은 에러나니 직접 다운받아 지정 필요).")
    ap.add_argument("--prepare-brush", action="store_true", help="COLMAP 이후 Brush 학습용 transforms.json 생성")
    ap.add_argument("--run-brush", action="store_true",
                    help="COLMAP 완료 후 Brush 학습까지 자동 실행 (--prepare-brush 자동 적용)")
    ap.add_argument("--brush-exe", default=None,
                    help="Brush 학습 실행파일 경로/이름 (기본: Windows는 brush.exe, macOS는 brush, PATH에 등록되어 있어야 함)")
    ap.add_argument("--brush-args", default=None,
                    help='Brush 실행 시 추가로 전달할 인자 문자열 (예: "--total-train-iters 30000 --export-every 5000")')
    args = ap.parse_args()

    video_paths = args.input
    out_dir = Path(args.output)
    raw_dir = out_dir / "_raw"
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_log_file(Path(video_paths[0]).resolve().parent / "log.txt")
    camera_model = None
    camera_params = None

    if args.start_from_brush:
        log("[i] --start-from-brush: 이미지 추출/COLMAP 없이 기존 colmap/ 폴더로 Brush 학습만 재시작합니다.")
        colmap_dir = out_dir / "colmap"
        try:
            brush_extra_args = shlex.split(args.brush_args) if args.brush_args else None
            run_brush_training(colmap_dir, brush_exe=args.brush_exe, extra_args=brush_extra_args)
            log("[i] Brush 학습 완료")
        except FileNotFoundError as e:
            log(f"[!] Brush 실행 준비 실패: {e}")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            log(f"[!] Brush 학습 실행 실패: {e}")
            sys.exit(1)
        log("[OK] 완료")
        return

    if not args.start_from_colmap:
        missing = [vp for vp in video_paths if not Path(vp).exists()]
        for vp in missing:
            log(f"[!] 입력 영상을 찾을 수 없습니다: {vp}")
        if missing:
            sys.exit(1)

    if args.start_from_colmap:
        log("[i] --start-from-colmap: 이미지 추출을 건너뛰고 COLMAP부터 시작합니다.")
        if not images_dir.exists() or not any(images_dir.iterdir()):
            log(f"[!] {images_dir}에 이미지가 없어 COLMAP을 시작할 수 없습니다. 먼저 프레임을 추출하거나 images/ 폴더를 준비하세요.")
            sys.exit(1)
        mode = "normal"
        failed_views = []
        kept, skipped_blur, skipped_person, total_raw = 0, 0, 0, 0
        video_results = []
    else:
        multi = len(video_paths) > 1
        if multi:
            log(f"[i] 영상 {len(video_paths)}편을 하나의 COLMAP 재구성으로 합쳐서 처리합니다.")

        video_results = []
        for idx, vp in enumerate(video_paths):
            prefix = f"v{idx}_" if multi else ""
            video_raw_dir = (raw_dir / f"v{idx}") if multi else raw_dir
            if multi:
                log(f"[i] ({idx + 1}/{len(video_paths)}) 영상 처리 시작: {vp}")
            video_results.append(
                process_single_video(vp, args, video_raw_dir, images_dir, masks_dir, filename_prefix=prefix)
            )
        shutil.rmtree(raw_dir, ignore_errors=True)

        modes = {r["mode"] for r in video_results}
        if len(modes) > 1:
            log(f"[!] 입력 영상들의 추출 모드가 서로 다릅니다({sorted(modes)}). 하나의 COLMAP 재구성으로 "
                "합치려면 --mode로 모든 영상에 같은 모드를 강제하고 다시 실행하세요.")
            sys.exit(1)
        mode = video_results[0]["mode"]

        camera_params_set = {r["camera_params"] for r in video_results if r["camera_params"] is not None}
        if len(camera_params_set) > 1:
            log("[!] 입력 영상들의 카메라 파라미터(--fov/--preset/--resize 등)가 서로 달라 COLMAP이 전체 "
                "이미지를 하나의 카메라로 취급할 수 없습니다. 모든 영상에 같은 추출 옵션을 사용하세요.")
            sys.exit(1)

        camera_model = video_results[0]["camera_model"]
        camera_params = video_results[0]["camera_params"]
        failed_views = [v for r in video_results for v in r["failed_views"]]
        kept = sum(r["kept"] for r in video_results)
        skipped_blur = sum(r["skipped_blur"] for r in video_results)
        skipped_person = sum(r["skipped_person"] for r in video_results)
        total_raw = sum(r["total_raw"] for r in video_results)

        if multi:
            log(f"[i] 전체 결과({len(video_paths)}개 영상 합산): 사용 {kept}장 / 블러 제외 {skipped_blur}장 / "
                f"사람과다 제외 {skipped_person}장")

    metadata = {
        "source_videos": [str(vp) for vp in video_paths],
        "mode": mode,
        "videos": [
            {
                "video_path": r["video_path"],
                "duration": r["duration"],
                "preset": r["preset"],
                "fov": r["fov"],
                "fps": r["fps"],
                "yaw_list": r["yaw_list"],
                "pitch_list": r["pitch_list"],
                "interval": r["interval"],
                "kept": r["kept"],
                "skipped_blur": r["skipped_blur"],
                "skipped_person": r["skipped_person"],
                "total_raw": r["total_raw"],
                "failed_views": r["failed_views"],
            }
            for r in video_results
        ],
        "masking_enabled": not args.no_mask,
        "blur_thresh": args.blur_thresh,
        "person_skip_ratio": args.person_skip_ratio,
        "failed_views": failed_views,
        "counts": {
            "kept": kept,
            "skipped_blur": skipped_blur,
            "skipped_person": skipped_person,
            "total_raw": total_raw,
        },
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir) if not args.no_mask else None,
        "frames": [f for r in video_results for f in r["metadata_frames"]],
    }
    meta_path = out_dir / "gs_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log(f"[i] 메타데이터 저장: {meta_path}")

    if args.run_colmap or args.start_from_colmap:
        db_path = args.colmap_db or str(out_dir / "database.db")
        matcher = args.colmap_matcher
        if matcher is None:
            if len(video_paths) > 1:
                # sequential matcher는 "파일명 순서 = 실제 촬영 경로 순서"라는 가정에
                # 의존한다. 영상 여러 개를 합치면 colmap_images_dir에서 v0_*가 v1_*보다
                # 앞에 오도록 정렬되는데, 두 영상은 서로 다른 촬영 경로이므로 그 경계에서
                # sequential matcher가 겹치는 프레임끼리 전혀 매칭을 시도하지 않는다.
                # 그 결과 영상별로 재구성이 쪼개져 버리므로, 모든 쌍을 확인하는
                # exhaustive matcher를 강제한다.
                matcher = "exhaustive"
                log(f"[i] --colmap-matcher 미지정: 영상 {len(video_paths)}개를 합치는 경우 "
                    "sequential matcher는 영상 경계에서 매칭이 끊기므로 'exhaustive' 자동 선택")
            else:
                # equirectangular는 서로 다른 yaw/pitch 뷰가 파일명 정렬 순서상 번갈아 오므로
                # sequential matcher로는 실제로 겹치는 뷰끼리 매칭되지 않아 재구성이 조각날 수 있다.
                matcher = "exhaustive" if mode == "equirectangular" else "sequential"
                log(f"[i] --colmap-matcher 미지정: 모드({mode})에 따라 '{matcher}' 자동 선택")
        try:
            colmap_result = run_colmap_pipeline(
                out_dir,
                images_dir,
                masks_dir,
                db_path,
                matcher=matcher,
                prepare_brush=args.prepare_brush or args.run_colmap,
                vocab_tree_path=args.vocab_tree_path,
                camera_model=camera_model,
                camera_params=camera_params,
            )
            log(f"[i] COLMAP 파이프라인 완료. 결과: {colmap_result['colmap_dir']}")
            if args.run_brush:
                try:
                    brush_extra_args = shlex.split(args.brush_args) if args.brush_args else None
                    # brush_dir(transforms.json)이 아니라 colmap_dir(images/+sparse/0/)을
                    # 넘겨야 Brush가 COLMAP 포인트클라우드로 가우시안을 초기화한다.
                    # transforms.json만 쓰면 랜덤 초기화로 시작해 품질이 크게 떨어진다.
                    run_brush_training(Path(colmap_result["colmap_dir"]), brush_exe=args.brush_exe, extra_args=brush_extra_args)
                    log("[i] Brush 학습 완료")
                except FileNotFoundError as e:
                    log(f"[!] Brush 실행 준비 실패: {e}")
                except subprocess.CalledProcessError as e:
                    log(f"[!] Brush 학습 실행 실패: {e}")
        except FileNotFoundError:
            log("[!] colmap 실행파일을 찾을 수 없습니다. PATH에 colmap이 등록되어 있는지 확인하세요.")
        except subprocess.CalledProcessError as e:
            log(f"[!] COLMAP 실행 실패: {e}")

    log("[OK] 완료")


if __name__ == "__main__":
    main()
