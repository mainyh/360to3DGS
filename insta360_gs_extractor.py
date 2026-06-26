#!/usr/bin/env python3
"""
insta360_gs_extractor.py
=========================
Insta360 360도(equirectangular) 영상 -> COLMAP/3DGS용 멀티뷰 프레임 추출 + 사람 마스킹 통합 스크립트

파이프라인:
  1) ffmpeg v360 필터로 equirectangular -> 여러 yaw/pitch 방향의 perspective 뷰 추출 (오버랩 60~75%)
  2) 블러(라플라시안 분산) 프레임 자동 제외
  3) YOLOv8-seg로 사람(person) 마스크 생성, COLMAP mask 컨벤션으로 저장
  4) 사람이 화면을 과도하게 가리는 프레임은 자동 스킵
  5) gs_metadata.json 작성 (COLMAP/3DGS 다음 단계에서 참조)
  6) (옵션) COLMAP feature_extractor를 mask_path 지정해서 바로 실행

사용 예:
  python insta360_gs_extractor.py ^
      --input "D:\\OneDrive.Golfzon\\Temp\\3DGS\\healingstay\\01.mp4" ^
      --output "D:\\Temp\\skysplat-test1\\healingstay" ^
      --preset indoor_dense ^
      --run-colmap

필요 패키지:
  pip install ultralytics opencv-python numpy
  ffmpeg, (옵션) colmap 이 PATH에 있어야 함
"""

import argparse
import json
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# 씬 프리셋: yaw 각도 리스트 / pitch / fov / 추출 fps
#   - overlap 60~75%를 만족하도록 yaw 간격과 fov를 설계
# ----------------------------------------------------------------------------
SCENE_PRESETS = {
    # 야외, 넓은 골프 코스 등 - 적은 방향, 넓은 fov
    "outdoor_wide": {
        "yaw_list": [0, 60, 120, 180, 240, 300],
        "pitch_list": [-10, 10],
        "fov": 100,
        "fps": 2,
        "out_w": 1920,
        "out_h": 1080,
    },
    # 실내, 사람 동선 있는 좁은 공간(healingstay 등) - 더 많은 방향, 좁은 fov, 높은 fps
    "indoor_dense": {
        "yaw_list": [0, 45, 90, 135, 180, 225, 270, 315],
        "pitch_list": [-15, 0, 15],
        "fov": 90,
        "fps": 3,
        "out_w": 1920,
        "out_h": 1080,
    },
    # 좁은 통로/계단 등 구조물
    "corridor": {
        "yaw_list": [0, 45, 90, 135, 180, 225, 270, 315],
        "pitch_list": [-20, 0],
        "fov": 80,
        "fps": 4,
        "out_w": 1920,
        "out_h": 1080,
    },
    # 단일 오브젝트(조형물, 그린 주변 등) 360 캡처
    "object_orbit": {
        "yaw_list": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        "pitch_list": [0],
        "fov": 70,
        "fps": 2,
        "out_w": 1920,
        "out_h": 1080,
    },
    # 빠른 프리뷰/테스트용 - 최소 방향
    "preview_fast": {
        "yaw_list": [0, 90, 180, 270],
        "pitch_list": [0],
        "fov": 100,
        "fps": 1,
        "out_w": 1280,
        "out_h": 720,
    },
}

PERSON_CLASS_ID = 0  # COCO: person


def run(cmd, **kwargs):
    print(f"[cmd] {' '.join(str(c) for c in cmd)}")
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        # 캡처된 stdout/stderr가 있으면 반드시 출력 (그래야 실제 원인을 알 수 있음)
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
    stream = data["streams"][0]
    return stream


def normalize_yaw(yaw: float) -> float:
    """ffmpeg v360 필터는 yaw를 -180~180 범위로만 받음. 0~360 입력을 정규화."""
    yaw = ((yaw + 180) % 360) - 180
    return yaw


def extract_view(video_path: str, out_dir: Path, yaw: float, pitch: float,
                  fov: float, fps: int, out_w: int, out_h: int, tag: str):
    """ffmpeg v360 필터로 equirectangular -> perspective(flat) 뷰 추출"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{tag}_%05d.jpg")

    yaw_norm = normalize_yaw(yaw)

    # v360: input=e (equirectangular), output=flat (perspective)
    # yaw/pitch/roll 단위 degree, h_fov/v_fov 단위 degree
    vf = (
        f"v360=e:flat:yaw={yaw_norm}:pitch={pitch}:roll=0:"
        f"h_fov={fov}:v_fov={fov}:w={out_w}:h={out_h},"
        f"fps={fps}"
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
    """
    사람 영역 마스크 생성.
    반환: (mask, person_area_ratio)
      mask: COLMAP 컨벤션 - 사람=0(검정, 무시), 배경=255(흰색, 사용)
      person_area_ratio: 전체 프레임 중 사람이 차지하는 비율 (0~1)
    """
    h, w = img.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)  # 기본: 전부 사용

    results = yolo_model.predict(img, classes=[PERSON_CLASS_ID], conf=conf, verbose=False)
    person_pixels = 0

    for r in results:
        if r.masks is None:
            continue
        for seg in r.masks.data.cpu().numpy():
            seg_resized = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
            person_area = seg_resized > 0.5
            mask[person_area] = 0
            person_pixels += int(person_area.sum())

    # 약간 팽창시켜서 경계 흐릿한 부분도 같이 제외(안전 마진)
    if person_pixels > 0:
        kernel = np.ones((9, 9), np.uint8)
        inv = cv2.bitwise_not(mask)
        inv = cv2.dilate(inv, kernel, iterations=1)
        mask = cv2.bitwise_not(inv)

    ratio = float((mask == 0).sum()) / (h * w)
    return mask, ratio


def report_blur_stats(raw_frames):
    """필터링 전에 전체 프레임의 블러 분산값 분포를 보여줌 (임계값 결정용)"""
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
    print(f"[i] 블러 분산값 통계 (n={len(arr)}): "
          f"min={arr.min():.1f}  p10={np.percentile(arr,10):.1f}  "
          f"median={np.median(arr):.1f}  p90={np.percentile(arr,90):.1f}  max={arr.max():.1f}")
    return variances


def process_frames(
    raw_frames,
    images_dir: Path,
    masks_dir: Path,
    yolo_model,
    blur_thresh: float,
    person_skip_ratio: float,
    person_conf: float,
):
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    kept, skipped_blur, skipped_person = 0, 0, 0
    metadata_frames = []

    for fp in raw_frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # 1) 블러 필터
        var = laplacian_variance(img)
        if var < blur_thresh:
            skipped_blur += 1
            fp.unlink(missing_ok=True)
            continue

        # 2) 사람 마스크
        mask, person_ratio = make_person_mask(yolo_model, img, conf=person_conf)

        # 3) 사람이 과도하게(임계 이상) 가린 프레임 스킵
        if person_ratio > person_skip_ratio:
            skipped_person += 1
            fp.unlink(missing_ok=True)
            continue

        # 4) 최종 저장: images/, masks/ (COLMAP 컨벤션: <image_name>.png)
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


def main():
    ap = argparse.ArgumentParser(description="Insta360 -> COLMAP/3DGS 멀티뷰 프레임 추출 + 사람 마스킹")
    ap.add_argument("--input", required=True, help="입력 360 영상 경로 (equirectangular)")
    ap.add_argument("--output", required=True, help="출력 디렉토리")
    ap.add_argument("--preset", default="indoor_dense", choices=SCENE_PRESETS.keys())
    ap.add_argument("--blur-thresh", type=float, default=15.0, help="라플라시안 분산 임계값(이하면 제외). 영상 특성에 따라 차이가 크니 첫 실행 로그의 통계를 보고 조정하세요.")
    ap.add_argument("--person-skip-ratio", type=float, default=0.35, help="사람이 이 비율 이상 가리면 프레임 스킵")
    ap.add_argument("--person-conf", type=float, default=0.3, help="YOLO person confidence threshold")
    ap.add_argument("--yolo-model", default="yolov8n-seg.pt")
    ap.add_argument("--no-mask", action="store_true", help="사람 마스킹 비활성화(블러 필터만)")
    ap.add_argument("--run-colmap", action="store_true", help="추출 후 colmap feature_extractor까지 자동 실행")
    ap.add_argument("--colmap-db", default=None, help="colmap database 경로 (기본: <output>/database.db)")
    args = ap.parse_args()

    video_path = args.input
    out_dir = Path(args.output)
    raw_dir = out_dir / "_raw"
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(video_path).exists():
        print(f"[!] 입력 영상을 찾을 수 없음: {video_path}")
        sys.exit(1)

    print(f"[i] 영상 정보 확인 중...")
    info = ffprobe_info(video_path)
    print(f"    해상도: {info['width']}x{info['height']}  fps: {info['r_frame_rate']}")
    aspect = float(info["width"]) / float(info["height"])
    if abs(aspect - 2.0) > 0.05:
        print(f"[!] 경고: 가로:세로 비율이 {aspect:.2f}:1 입니다. equirectangular(2:1)가 아닐 수 있습니다.")
        print("    Insta360 Studio에서 '스티칭+내보내기'를 equirectangular로 했는지 확인하세요.")

    preset = SCENE_PRESETS[args.preset]
    print(f"[i] 프리셋 '{args.preset}' 적용: yaw {len(preset['yaw_list'])}방향 x pitch {len(preset['pitch_list'])}각도")

    # 1) 멀티뷰 추출
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_raw_frames = []
    failed_views = []
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
                print(f"[!] yaw={yaw} pitch={pitch} 1차 실패. 2초 대기 후 재시도...")
                time.sleep(2)
                try:
                    frames = extract_view(
                        video_path, raw_dir, yaw, pitch,
                        preset["fov"], preset["fps"], preset["out_w"], preset["out_h"], tag,
                    )
                    all_raw_frames.extend(frames)
                except subprocess.CalledProcessError:
                    print(f"[!] yaw={yaw} pitch={pitch} 재시도도 실패 -> 이 뷰는 스킵하고 계속 진행합니다.")
                    failed_views.append({"yaw": yaw, "pitch": pitch})
                    continue
    print(f"[i] 총 원본 프레임 {len(all_raw_frames)}장 추출 완료 ({time.time()-t0:.1f}s)")
    if failed_views:
        print(f"[!] 실패한 뷰 {len(failed_views)}개: {failed_views}")
        print("    위 stderr 로그를 확인해서 원인(코덱/메모리/디코더 충돌 등)을 파악하세요.")

    # 2) 블러 + 사람 마스킹 처리
    if args.no_mask:
        print("[i] --no-mask: 사람 마스킹 생략, 블러 필터만 적용")
        var_stats = report_blur_stats(all_raw_frames)
        images_dir.mkdir(parents=True, exist_ok=True)
        kept, skipped_blur, metadata_frames = 0, 0, []
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
        print(f"[!] 경고: 블러 필터로 모든 프레임이 제거되었습니다 (blur_thresh={args.blur_thresh}).")
        print(f"    이 영상의 분산값 분포상 --blur-thresh {suggested:.1f} 정도로 낮춰서 다시 실행해보세요.")
        print(f"    예: python insta360_gs_extractor.py ... --blur-thresh {suggested:.1f}")

    shutil.rmtree(raw_dir, ignore_errors=True)

    print(f"[i] 결과: 사용 {kept}장 / 블러 제외 {skipped_blur}장 / 사람과다 제외 {skipped_person}장")

    # 3) gs_metadata.json
    metadata = {
        "source_video": str(video_path),
        "preset": args.preset,
        "preset_params": preset,
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

    # 4) COLMAP 자동 실행 (옵션)
    if args.run_colmap:
        db_path = args.colmap_db or str(out_dir / "database.db")
        print(f"[i] COLMAP feature_extractor 실행 중...")
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", db_path,
            "--image_path", str(images_dir),
        ]
        if not args.no_mask:
            cmd += ["--ImageReader.mask_path", str(masks_dir)]
        try:
            run(cmd)
            print(f"[i] COLMAP feature_extractor 완료. database: {db_path}")
            print("[i] 다음 단계: colmap sequential_matcher / mapper 를 이어서 실행하세요.")
        except FileNotFoundError:
            print("[!] colmap 실행파일을 찾을 수 없습니다. PATH에 colmap이 등록되어 있는지 확인하세요.")
        except subprocess.CalledProcessError as e:
            print(f"[!] COLMAP 실행 실패: {e}")

    print("[✓] 완료")


if __name__ == "__main__":
    main()