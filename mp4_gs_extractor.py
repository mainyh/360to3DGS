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


def run(cmd, **kwargs):
    print(f"[cmd] {' '.join(str(c) for c in cmd)}")
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print("---- stdout ----")
            print(e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="replace"))
        if e.stderr:
            print("---- stderr ----")
            print(e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace"))
        raise


def ffprobe_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    return data["streams"][0]


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


def is_equirectangular(stream: dict, sample_info: dict | None = None) -> bool:
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


def extract_normal_frames(video_path: str, out_dir: Path, timestamps, out_w: int, out_h: int, keep_aspect: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for idx, ts in enumerate(timestamps, 1):
        fname = f"T{ts:07.3f}.jpg"
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
            print(f"[!] t={ts}s 프레임 추출 실패, 건너뜁니다 ({idx}/{len(timestamps)})")
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


def extract_view(video_path: str, out_dir: Path, yaw: float, pitch: float, fov: float, fps: int, out_w: int, out_h: int, tag: str):
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
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-q:v", "2",
        pattern,
    ]
    run(cmd, capture_output=True, text=True)
    return sorted(out_dir.glob(f"{tag}_*.jpg"))


def laplacian_variance(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_yolo_seg(model_name: str = "yolov8n-seg.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics 미설치. 설치: pip install ultralytics")
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
        print("[!] 읽을 수 있는 프레임이 없습니다.")
        return []
    arr = np.array(variances)
    print(f"[i] 블러 분산값 통계 (n={len(arr)}): min={arr.min():.1f}  p10={np.percentile(arr,10):.1f}  median={np.median(arr):.1f}  p90={np.percentile(arr,90):.1f}  max={arr.max():.1f}")
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
        print(f"[i] 이미지 소스 디렉토리 자동 감지: {source_dir}")

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


def run_colmap_pipeline(out_dir: Path, images_dir: Path, masks_dir: Path, db_path: str, matcher: str = "sequential", prepare_brush: bool = False, vocab_tree_path: str = None):
    colmap_dir = out_dir / "colmap"
    colmap_images_dir = colmap_dir / "images"
    colmap_masks_dir = colmap_dir / "masks"
    sparse_dir = colmap_dir / "sparse"
    export_dir = colmap_dir / "export"
    brush_dir = out_dir / "brush_prepared"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    prepare_colmap_inputs(images_dir, masks_dir, colmap_images_dir, colmap_masks_dir)
    print(f"[i] COLMAP 입력 이미지 정렬 완료: {colmap_images_dir}")

    feature_cmd = [
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", str(colmap_images_dir),
    ]
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

    export_dir.mkdir(parents=True, exist_ok=True)
    converter_cmd = [
        "colmap", "model_converter",
        "--input_path", str(sparse_dir / "0"),
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
            print(f"[i] Brush 전처리 완료: {brush_dir}")
        else:
            print("[!] COLMAP sparse 모델을 찾지 못해 Brush 전처리를 건너뜁니다.")

    return {
        "colmap_dir": str(colmap_dir),
        "sparse_dir": str(sparse_dir),
        "export_dir": str(export_dir),
        "brush_dir": str(brush_dir),
    }


def run_brush_training(dataset_dir: Path, brush_exe: str = "brush.exe", extra_args=None):
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
        print("[!] sparse/*/points3D.*가 없어 포인트클라우드 초기화 없이(랜덤 초기화) 학습합니다. "
              "COLMAP 결과 폴더(images/ + sparse/0/)를 넘기면 품질이 크게 좋아집니다.")
    cmd = [brush_exe, str(dataset_dir)]
    if extra_args:
        cmd.extend(extra_args)
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description="MP4 영상에서 COLMAP/3DGS용 이미지 추출 + 사람 마스킹")
    ap.add_argument("--input", required=True, help="입력 MP4 영상 경로")
    ap.add_argument("--output", required=True, help="출력 디렉토리")
    ap.add_argument("--mode", choices=["auto", "normal", "equirectangular"], default="auto",
                    help="auto: 영상 샘플 프레임과 해상도 기준으로 일반/360 자동 선택, normal: 일반 MP4 추출, equirectangular: 360 추출")
    ap.add_argument("--preset", default="indoor_dense", choices=SCENE_PRESETS.keys(),
                    help="360 추출일 때 사용할 프리셋")
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
    ap.add_argument("--colmap-matcher", default="sequential", choices=["sequential", "exhaustive", "vocab_tree"], help="COLMAP matcher 종류")
    ap.add_argument("--vocab-tree-path", default=None,
                    help="vocab_tree matcher용 사전학습 vocabulary tree(.bin) 경로. "
                         "지정 안 하면 최신 COLMAP은 자동 다운로드/캐싱을 시도함 "
                         "(구버전 COLMAP은 에러나니 직접 다운받아 지정 필요).")
    ap.add_argument("--prepare-brush", action="store_true", help="COLMAP 이후 Brush 학습용 transforms.json 생성")
    ap.add_argument("--run-brush", action="store_true",
                    help="COLMAP 완료 후 Brush 학습까지 자동 실행 (--prepare-brush 자동 적용)")
    ap.add_argument("--brush-exe", default="brush.exe",
                    help="Brush 학습 실행파일 경로/이름 (기본: brush.exe, PATH에 등록되어 있어야 함)")
    ap.add_argument("--brush-args", default=None,
                    help='Brush 실행 시 추가로 전달할 인자 문자열 (예: "--total-train-iters 30000 --export-every 5000")')
    args = ap.parse_args()

    video_path = args.input
    out_dir = Path(args.output)
    raw_dir = out_dir / "_raw"
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.start_from_brush:
        print("[i] --start-from-brush: 이미지 추출/COLMAP 없이 기존 colmap/ 폴더로 Brush 학습만 재시작합니다.")
        colmap_dir = out_dir / "colmap"
        try:
            brush_extra_args = shlex.split(args.brush_args) if args.brush_args else None
            run_brush_training(colmap_dir, brush_exe=args.brush_exe, extra_args=brush_extra_args)
            print("[i] Brush 학습 완료")
        except FileNotFoundError as e:
            print(f"[!] Brush 실행 준비 실패: {e}")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"[!] Brush 학습 실행 실패: {e}")
            sys.exit(1)
        print("[OK] 완료")
        return

    if not args.start_from_colmap and not Path(video_path).exists():
        print(f"[!] 입력 영상을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    if args.start_from_colmap:
        print("[i] --start-from-colmap: 이미지 추출을 건너뛰고 COLMAP부터 시작합니다.")
        if not images_dir.exists() or not any(images_dir.iterdir()):
            print(f"[!] {images_dir}에 이미지가 없어 COLMAP을 시작할 수 없습니다. 먼저 프레임을 추출하거나 images/ 폴더를 준비하세요.")
            sys.exit(1)
        mode = "normal"
        duration = 0.0
        all_raw_frames = []
        failed_views = []
        kept, skipped_blur, skipped_person, metadata_frames = 0, 0, 0, []
        var_stats = []
    else:
        print("[i] 영상 정보 확인 중...")
        info = ffprobe_info(video_path)
        width = int(info["width"])
        height = int(info["height"])
        duration = float(info.get("duration", 0.0))
        aspect = float(width) / float(height)
        print(f"    해상도: {width}x{height}  fps: {info['r_frame_rate']}  길이: {duration:.2f}s")

        sample_info = inspect_video_sample(video_path)
        phone_note = " (스마트폰 영상 감지)" if sample_info.get("is_smartphone") else ""
        print(f"    샘플 프레임: {sample_info['width']}x{sample_info['height']} aspect={sample_info['aspect']:.2f}{phone_note}")

        if args.mode == "auto":
            if is_equirectangular(info, sample_info):
                mode = "equirectangular"
                print("[i] 자동 판별: 360°/equirectangular 영상으로 감지되어 360 추출 파이프라인을 사용합니다.")
            else:
                mode = "normal"
                print("[i] 자동 판별: 일반 카메라 영상으로 감지되어 시간 간격 추출을 사용합니다.")
        else:
            mode = args.mode
            print(f"[i] 강제 모드: {mode}")

        raw_dir.mkdir(parents=True, exist_ok=True)
        all_raw_frames = []
        failed_views = []

        if mode == "normal":
            interval = args.interval if args.interval else recommend_interval(duration)
            out_w, out_h = parse_resolution(args.resize)
            print(f"[i] 일반 MP4 추출: interval={interval}s start={args.start_time}s trim_end={args.trim_end}s resize={out_w}x{out_h}")
            timestamps = build_timestamps(duration, interval, args.start_time, args.trim_end)
            print(f"[i] 타임스탬프 수: {len(timestamps)}")
            all_raw_frames = extract_normal_frames(
                video_path, raw_dir, timestamps, out_w, out_h,
                keep_aspect=(args.resize_mode == "fit"),
            )
        else:
            preset = SCENE_PRESETS[args.preset]
            computed_v_fov = compute_square_pixel_fov(preset["fov"], preset["out_w"], preset["out_h"])
            print(f"[i] 360 추출 프리셋 '{args.preset}' 적용: yaw {len(preset['yaw_list'])} x pitch {len(preset['pitch_list'])}")
            print(f"    h_fov={preset['fov']}° -> 자동계산 v_fov={computed_v_fov:.2f}° (출력 {preset['out_w']}x{preset['out_h']}, 정사각형 픽셀 보장)")
            t0 = time.time()
            for yaw in preset["yaw_list"]:
                for pitch in preset["pitch_list"]:
                    tag = f"y{yaw:03d}_p{pitch:+03d}"
                    print(f"[i] 추출 중: yaw={yaw} pitch={pitch}")
                    try:
                        frames = extract_view(
                            video_path, raw_dir, yaw, pitch,
                            preset["fov"], preset["fps"], preset["out_w"], preset["out_h"], tag,
                        )
                        all_raw_frames.extend(frames)
                    except subprocess.CalledProcessError:
                        print(f"[!] yaw={yaw} pitch={pitch} 실패, 2초 대기 후 재시도")
                        time.sleep(2)
                        try:
                            frames = extract_view(
                                video_path, raw_dir, yaw, pitch,
                                preset["fov"], preset["fps"], preset["out_w"], preset["out_h"], tag,
                            )
                            all_raw_frames.extend(frames)
                        except subprocess.CalledProcessError:
                            print(f"[!] 재시도도 실패: yaw={yaw} pitch={pitch} -> 스킵")
                            failed_views.append({"yaw": yaw, "pitch": pitch})
            print(f"[i] 총 원본 프레임 {len(all_raw_frames)}장 추출 완료 ({time.time() - t0:.1f}s)")

        if args.no_mask:
            print("[i] --no-mask: 사람 마스킹 생략, 블러 필터만 적용")
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
            skipped_person = 0
        else:
            var_stats = report_blur_stats(all_raw_frames)
            print(f"[i] YOLO 모델 로드: {args.yolo_model}")
            yolo_model = load_yolo_seg(args.yolo_model)
            print("[i] 블러 필터 + 사람 마스킹 처리 중...")
            kept, skipped_blur, skipped_person, metadata_frames = process_frames(
                all_raw_frames, images_dir, masks_dir, yolo_model,
                args.blur_thresh, args.person_skip_ratio, args.person_conf,
            )

        if kept == 0 and var_stats:
            arr = np.array(var_stats)
            suggested = max(1.0, float(np.percentile(arr, 20)))
            print(f"[!] 블러 필터로 모든 프레임이 제거되었습니다 (blur_thresh={args.blur_thresh}).")
            print(f"    --blur-thresh {suggested:.1f} 정도로 낮춰서 다시 실행해보세요.")

        shutil.rmtree(raw_dir, ignore_errors=True)

        print(f"[i] 결과: 사용 {kept}장 / 블러 제외 {skipped_blur}장 / 사람과다 제외 {skipped_person}장")

    metadata = {
        "source_video": str(video_path),
        "mode": mode,
        "preset": args.preset if mode == "equirectangular" else None,
        "normal_params": {
            "interval": args.interval,
            "start_time": args.start_time,
            "trim_end": args.trim_end,
            "resize": args.resize,
            "resize_mode": args.resize_mode,
        } if mode == "normal" else None,
        "masking_enabled": not args.no_mask,
        "blur_thresh": args.blur_thresh,
        "person_skip_ratio": args.person_skip_ratio,
        "failed_views": failed_views,
        "counts": {
            "kept": kept,
            "skipped_blur": skipped_blur,
            "skipped_person": skipped_person,
            "total_raw": len(all_raw_frames),
        },
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir) if not args.no_mask else None,
        "frames": metadata_frames,
    }
    meta_path = out_dir / "gs_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[i] 메타데이터 저장: {meta_path}")

    if args.run_colmap or args.start_from_colmap:
        db_path = args.colmap_db or str(out_dir / "database.db")
        try:
            colmap_result = run_colmap_pipeline(
                out_dir,
                images_dir,
                masks_dir,
                db_path,
                matcher=args.colmap_matcher,
                prepare_brush=args.prepare_brush or args.run_colmap,
                vocab_tree_path=args.vocab_tree_path,
            )
            print(f"[i] COLMAP 파이프라인 완료. 결과: {colmap_result['colmap_dir']}")
            if args.run_brush:
                try:
                    brush_extra_args = shlex.split(args.brush_args) if args.brush_args else None
                    # brush_dir(transforms.json)이 아니라 colmap_dir(images/+sparse/0/)을
                    # 넘겨야 Brush가 COLMAP 포인트클라우드로 가우시안을 초기화한다.
                    # transforms.json만 쓰면 랜덤 초기화로 시작해 품질이 크게 떨어진다.
                    run_brush_training(Path(colmap_result["colmap_dir"]), brush_exe=args.brush_exe, extra_args=brush_extra_args)
                    print("[i] Brush 학습 완료")
                except FileNotFoundError as e:
                    print(f"[!] Brush 실행 준비 실패: {e}")
                except subprocess.CalledProcessError as e:
                    print(f"[!] Brush 학습 실행 실패: {e}")
        except FileNotFoundError:
            print("[!] colmap 실행파일을 찾을 수 없습니다. PATH에 colmap이 등록되어 있는지 확인하세요.")
        except subprocess.CalledProcessError as e:
            print(f"[!] COLMAP 실행 실패: {e}")

    print("[OK] 완료")


if __name__ == "__main__":
    main()