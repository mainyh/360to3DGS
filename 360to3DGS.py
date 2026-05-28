"""
insta360_gs_extractor.py
────────────────────────────────────────────────────────────────────────────
인스타360 360° 영상 → 가우시안 스플래팅용 멀티뷰 이미지 추출기

원리
  - 등장방형(equirectangular) 영상을 ffmpeg v360 필터로 flat 투영 분할
  - Pitch × Yaw 격자를 좌상→우하 순서로 순회 (COLMAP이 선호하는 연속성)
  - 인접 뷰 간 60~70% 중첩 → 특징점 매칭 품질 극대화
  - 시간 간격: 장면 속도에 따라 자동 권장 / 수동 지정 가능

사용법
  python insta360_gs_extractor.py <video.mp4> [옵션]

예시
  python insta360_gs_extractor.py D:/Temp/skysplat-test1/footage.mp4
  python insta360_gs_extractor.py footage.mp4 --interval 0.5 --fov 90 --out my_frames
  python insta360_gs_extractor.py footage.mp4 --preset indoor
  python insta360_gs_extractor.py footage.mp4 --no-time  # 각도 추출만, 시간 샘플링 없음
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 구조
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ViewConfig:
    """단일 뷰포인트 (시간 + 카메라 각도)"""
    frame_index: int        # 전체 추출 순서 번호
    timestamp: float        # 초 단위
    pitch: float            # 상하 각도 (-90 ~ +90)
    yaw: float              # 좌우 각도 (-180 ~ +180)
    row: int                # 격자 행 (위→아래)
    col: int                # 격자 열 (좌→우)


@dataclass
class ExtractionPlan:
    """추출 계획 전체"""
    video_path: str
    duration: float
    fps: float
    width: int
    height: int
    timestamps: List[float]
    pitches: List[float]
    yaws: List[float]
    fov: float
    output_size: int
    views: List[ViewConfig] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return len(self.timestamps) * len(self.pitches) * len(self.yaws)


# ══════════════════════════════════════════════════════════════════════════════
# 프리셋 정의
# ══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    "outdoor": {
        "description": "야외 넓은 공간 (건물 외관, 공원, 광장)",
        "fov": 100,
        "pitch_levels": 3,
        "pitch_range": (-30, 30),
        "yaw_steps": 12,
        "interval": 1.0,
        "output_size": 1024,
        "overlap_target": 0.65,
    },
    "indoor": {
        "description": "실내 공간 (방, 사무실, 카페)",
        "fov": 90,
        "pitch_levels": 3,
        "pitch_range": (-25, 25),
        "yaw_steps": 10,
        "interval": 0.5,
        "output_size": 1024,
        "overlap_target": 0.70,
    },
    "landscape": {
        "description": "경관 / 풍경 (산, 바다, 넓은 야외)",
        "fov": 110,
        "pitch_levels": 2,
        "pitch_range": (-20, 20),
        "yaw_steps": 8,
        "interval": 2.0,
        "output_size": 1024,
        "overlap_target": 0.60,
    },
    "object": {
        "description": "소형 오브젝트 촬영 (제품, 조각, 모형)",
        "fov": 75,
        "pitch_levels": 4,
        "pitch_range": (-45, 45),
        "yaw_steps": 16,
        "interval": 0.3,
        "output_size": 1024,
        "overlap_target": 0.75,
    },
    "fastmove": {
        "description": "빠른 이동 촬영 (드론, 달리기, 차량)",
        "fov": 95,
        "pitch_levels": 2,
        "pitch_range": (-20, 15),
        "yaw_steps": 10,
        "interval": 0.25,
        "output_size": 1024,
        "overlap_target": 0.65,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

EXEC_ENV = {
    "ffmpeg": "FFMPEG_PATH",
    "ffprobe": "FFPROBE_PATH",
}
EXEC_OVERRIDES = {
    "ffmpeg": None,
    "ffprobe": None,
}


def resolve_executable(name: str, explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        explicit = Path(explicit_path)
        if explicit.exists():
            return str(explicit)
        raise FileNotFoundError(
            f"지정한 {name} 경로를 찾을 수 없습니다: {explicit_path}. "
            f"올바른 경로를 입력하거나 환경 변수 {EXEC_ENV[name]}를 설정하세요."
        )

    env_path = os.environ.get(EXEC_ENV[name])
    if env_path:
        explicit = Path(env_path)
        if explicit.exists():
            return str(explicit)
        raise FileNotFoundError(
            f"환경 변수 {EXEC_ENV[name]}에 지정된 경로를 찾을 수 없습니다: {env_path}."
        )

    path = shutil.which(name)
    if path:
        return path

    raise FileNotFoundError(
        f"필수 실행 파일을 찾을 수 없습니다: {name}. "
        f"'{name}'를 설치하고 시스템 PATH에 추가하거나, 환경 변수 {EXEC_ENV[name]}를 설정하세요."
    )


def run(cmd: List[str], capture=True) -> subprocess.CompletedProcess:
    if cmd and cmd[0] in EXEC_ENV:
        cmd[0] = resolve_executable(cmd[0], EXEC_OVERRIDES[cmd[0]])
    try:
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"필수 실행 파일을 찾을 수 없습니다: {cmd[0]}. "
            f"'{cmd[0]}'를 설치하고 시스템 PATH에 추가하거나, 설치 경로를 확인하세요."
        ) from exc


def probe_video(path: str) -> Tuple[float, float, int, int]:
    """ffprobe로 영상 메타데이터 추출 → (duration, fps, width, height)"""
    probe = run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path
    ])
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe 실패:\n{probe.stderr}")

    info = json.loads(probe.stdout)
    video_stream = next(
        s for s in info["streams"] if s.get("codec_type") == "video"
    )

    duration = float(info["format"]["duration"])

    # FPS 파싱 (분수 형태 "60000/1001" 등 처리)
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    return duration, fps, width, height


def compute_yaw_angles(fov: float, steps: int, overlap: float) -> List[float]:
    """
    중첩률을 고려한 yaw 각도 배열 생성.
    step_angle = fov * (1 - overlap)
    steps 수에 맞게 -180 ~ +180 내에서 균등 배치.
    """
    step_angle = fov * (1.0 - overlap)
    # 요청한 steps와 step_angle로 실제 커버 범위 계산
    total_span = step_angle * steps
    if total_span > 360:
        # 360도를 steps로 나눠 재계산
        step_angle = 360.0 / steps

    start = -180.0
    return [round(start + i * step_angle, 1) for i in range(steps)]


def compute_pitch_angles(levels: int, pitch_range: Tuple[float, float]) -> List[float]:
    """위→아래 순서 pitch 배열 생성"""
    lo, hi = pitch_range
    if levels == 1:
        return [0.0]
    step = (hi - lo) / (levels - 1)
    # 위(hi)에서 아래(lo) 순서로
    return [round(hi - i * step, 1) for i in range(levels)]


def build_timestamps(duration: float, interval: float,
                     trim_start: float = 0.0, trim_end: float = 0.0) -> List[float]:
    """시간 샘플링 포인트 생성"""
    start = trim_start
    end = duration - trim_end
    if start >= end:
        raise ValueError("trim_start + trim_end >= 영상 길이")
    ts = []
    t = start
    while t <= end + 1e-9:
        ts.append(round(t, 3))
        t += interval
    return ts


def estimate_overlap(fov: float, yaw_step: float) -> float:
    """실제 중첩률 계산"""
    return max(0.0, 1.0 - yaw_step / fov)


def recommend_interval(duration: float, motion_speed: str = "normal") -> float:
    """
    영상 길이 + 이동 속도에 따른 권장 시간 간격.
    가우시안 스플래팅은 보통 100~300장의 시간 프레임이 적절.
    """
    targets = {"slow": 150, "normal": 100, "fast": 60}
    target_frames = targets.get(motion_speed, 100)
    interval = duration / target_frames
    # 0.1s ~ 3.0s 범위로 클램핑
    return round(max(0.1, min(3.0, interval)), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 추출 계획 생성
# ══════════════════════════════════════════════════════════════════════════════

def build_plan(args) -> ExtractionPlan:
    print(f"\n{'═'*58}")
    print("  인스타360 → 가우시안 스플래팅 이미지 추출기")
    print(f"{'═'*58}")

    # ── 영상 프로브
    print(f"\n[1/4] 영상 분석 중: {args.video}")
    duration, fps, width, height = probe_video(args.video)
    print(f"      길이: {duration:.2f}초  FPS: {fps:.2f}  해상도: {width}×{height}")

    if width < height * 1.5:
        print("  ⚠  가로:세로 비율이 2:1이 아닙니다.")
        print("     인스타360 원본 등장방형 영상인지 확인하세요.")

    # ── 프리셋 적용
    preset = PRESETS.get(args.preset, PRESETS["outdoor"])
    if args.preset:
        print(f"\n[2/4] 프리셋 적용: [{args.preset}] {preset['description']}")

    fov         = args.fov          or preset["fov"]
    pitch_levels = args.pitch_levels or preset["pitch_levels"]
    pitch_lo    = args.pitch_range[0] if args.pitch_range else preset["pitch_range"][0]
    pitch_hi    = args.pitch_range[1] if args.pitch_range else preset["pitch_range"][1]
    yaw_steps   = args.yaw_steps    or preset["yaw_steps"]
    overlap     = args.overlap      or preset["overlap_target"]
    out_size    = args.size         or preset["output_size"]

    # ── 각도 배열 계산
    print(f"\n[2/4] 뷰 격자 계산")
    pitches = compute_pitch_angles(pitch_levels, (pitch_lo, pitch_hi))
    yaws    = compute_yaw_angles(fov, yaw_steps, overlap)

    yaw_step    = abs(yaws[1] - yaws[0]) if len(yaws) > 1 else fov
    actual_overlap = estimate_overlap(fov, yaw_step)

    print(f"      FOV: {fov}°  출력: {out_size}×{out_size}px")
    print(f"      Pitch 레벨: {pitch_levels}개 {pitches}")
    print(f"      Yaw 스텝:   {yaw_steps}개 (간격 {yaw_step:.1f}°, 중첩 {actual_overlap*100:.0f}%)")
    print(f"      총 뷰 방향: {len(pitches) * len(yaws)}개")

    # ── 시간 샘플링
    print(f"\n[3/4] 시간 샘플링 계획")
    if args.no_time:
        timestamps = [0.0]
        print("      단일 타임스탬프 모드 (--no-time)")
    else:
        if args.interval:
            interval = args.interval
        else:
            interval = preset.get("interval", recommend_interval(duration))
            print(f"      권장 간격: {interval}초 (프리셋 기반)")
        timestamps = build_timestamps(
            duration, interval,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
        )
        print(f"      간격: {interval}초  →  {len(timestamps)}개 타임스탬프")
        print(f"      범위: {timestamps[0]:.2f}s ~ {timestamps[-1]:.2f}s")

    total = len(timestamps) * len(pitches) * len(yaws)
    print(f"\n      ★ 총 추출 이미지: {total}장")
    if total > 3000:
        print(f"  ⚠  3000장 초과! 시간 간격(--interval)을 늘리거나")
        print(f"     뷰 수(--yaw-steps, --pitch-levels)를 줄여보세요.")

    # ── 뷰 목록 조립 (좌상→우하 순서)
    views = []
    idx = 0
    for ts in timestamps:
        for r, pitch in enumerate(pitches):     # 위 → 아래
            for c, yaw in enumerate(yaws):      # 좌 → 우
                views.append(ViewConfig(
                    frame_index=idx,
                    timestamp=ts,
                    pitch=pitch,
                    yaw=yaw,
                    row=r,
                    col=c,
                ))
                idx += 1

    return ExtractionPlan(
        video_path=args.video,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        timestamps=timestamps,
        pitches=pitches,
        yaws=yaws,
        fov=fov,
        output_size=out_size,
        views=views,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ffmpeg 실행
# ══════════════════════════════════════════════════════════════════════════════

def extract_images(plan: ExtractionPlan, out_dir: Path, dry_run: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(plan.views)
    print(f"\n[4/4] 추출 시작 → {out_dir}")
    print(f"      총 {total}장  (Ctrl+C 로 중단 가능)\n")

    errors = []

    for i, v in enumerate(plan.views, 1):
        # 파일명: T초_P각도_Y각도.jpg  (숫자 정렬을 위해 패딩)
        fname = (
            f"T{v.timestamp:07.3f}_"
            f"R{v.row:02d}C{v.col:02d}_"
            f"P{v.pitch:+.0f}_"
            f"Y{v.yaw:+.0f}.jpg"
        )
        out_path = out_dir / fname

        if out_path.exists():
            print(f"  [{i:5d}/{total}] 건너뜀 (이미 존재): {fname}")
            continue

        # ffmpeg 커맨드 조립
        cmd = [
            "ffmpeg",
            "-ss", str(v.timestamp),    # 탐색 위치
            "-i", plan.video_path,
            "-vframes", "1",            # 1프레임만
            "-vf", (
                f"v360=e:flat:"
                f"yaw={v.yaw}:"
                f"pitch={v.pitch}:"
                f"h_fov={plan.fov}:"
                f"v_fov={plan.fov},"
                f"scale={plan.output_size}:{plan.output_size}"
            ),
            "-pix_fmt", "yuvj420p",
            "-q:v", "1",               # 최고 품질 JPEG
            str(out_path),
            "-loglevel", "error",
            "-y",
        ]

        progress = f"[{i:5d}/{total}] {v.timestamp:.2f}s P{v.pitch:+.0f}° Y{v.yaw:+.0f}°"

        if dry_run:
            print(f"  DRY  {progress}  →  {fname}")
            continue

        result = run(cmd)

        if result.returncode != 0:
            print(f"  ERR  {progress}  →  {result.stderr.strip()[:80]}")
            errors.append((fname, result.stderr.strip()))
        else:
            print(f"  OK   {progress}  →  {fname}")

    print(f"\n{'═'*58}")
    if dry_run:
        print(f"  DRY-RUN 완료. 실제 추출하려면 --dry-run 제거.")
    else:
        ok_count = total - len(errors)
        print(f"  완료: {ok_count}/{total}장 성공", end="")
        if errors:
            print(f"  /  실패: {len(errors)}장")
            for fname, msg in errors[:5]:
                print(f"    - {fname}: {msg}")
        else:
            print()
    print(f"  출력 폴더: {out_dir.resolve()}")
    print(f"{'═'*58}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 카메라 파라미터 내보내기 (COLMAP 힌트용)
# ══════════════════════════════════════════════════════════════════════════════

def export_metadata(plan: ExtractionPlan, out_dir: Path):
    """추출 계획을 JSON으로 저장 → COLMAP 커스텀 초기화 등에 활용"""
    meta = {
        "video": plan.video_path,
        "duration_sec": plan.duration,
        "fps": plan.fps,
        "source_resolution": [plan.width, plan.height],
        "fov_deg": plan.fov,
        "output_size_px": plan.output_size,
        "pitch_angles": plan.pitches,
        "yaw_angles": plan.yaws,
        "timestamps": plan.timestamps,
        "total_images": plan.total_images,
        "views": [
            {
                "file": (
                    f"T{v.timestamp:07.3f}_"
                    f"R{v.row:02d}C{v.col:02d}_"
                    f"P{v.pitch:+.0f}_"
                    f"Y{v.yaw:+.0f}.jpg"
                ),
                "timestamp": v.timestamp,
                "pitch_deg": v.pitch,
                "yaw_deg": v.yaw,
            }
            for v in plan.views
        ],
    }
    meta_path = out_dir / "gs_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  메타데이터 저장: {meta_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="인스타360 영상 → 가우시안 스플래팅용 멀티뷰 이미지 추출",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
프리셋 목록:
  outdoor   야외 넓은 공간 (기본값)
  indoor    실내 공간
  landscape 경관/풍경
  object    소형 오브젝트
  fastmove  빠른 이동 촬영

예시:
  python insta360_gs_extractor.py footage.mp4
  python insta360_gs_extractor.py footage.mp4 --preset indoor
  python insta360_gs_extractor.py footage.mp4 --interval 0.5 --fov 90
  python insta360_gs_extractor.py footage.mp4 --dry-run
        """
    )

    parser.add_argument("video",
        help="입력 MP4 경로")

    parser.add_argument("--preset", default="outdoor",
        choices=list(PRESETS.keys()),
        help="촬영 환경 프리셋 (기본: outdoor)")

    parser.add_argument("--out", default=None,
        help="출력 폴더 (기본: <영상이름>_GS_Frames)")

    # 시간 옵션
    parser.add_argument("--interval", type=float, default=None,
        help="시간 샘플링 간격(초). 미지정시 프리셋 값 사용")
    parser.add_argument("--trim-start", type=float, default=0.5,
        help="앞부분 자르기(초, 기본 0.5)")
    parser.add_argument("--trim-end", type=float, default=0.5,
        help="뒷부분 자르기(초, 기본 0.5)")
    parser.add_argument("--no-time", action="store_true",
        help="시간 축 샘플링 없이 0초만 사용 (각도 격자만 추출)")

    # 각도 옵션
    parser.add_argument("--fov", type=float, default=None,
        help="수평/수직 FOV 각도(°, 기본: 프리셋)")
    parser.add_argument("--yaw-steps", type=int, default=None,
        help="수평 분할 수 (기본: 프리셋)")
    parser.add_argument("--pitch-levels", type=int, default=None,
        help="수직 분할 수 (기본: 프리셋)")
    parser.add_argument("--pitch-range", type=float, nargs=2,
        metavar=("LO", "HI"), default=None,
        help="피치 범위(°) ex) --pitch-range -30 30")
    parser.add_argument("--overlap", type=float, default=None,
        help="인접 뷰 중첩률 0~1 (기본: 프리셋, 권장 0.60~0.75)")

    # 출력 옵션
    parser.add_argument("--size", type=int, default=None,
        help="출력 이미지 한 변 픽셀 (기본: 1024)")
    parser.add_argument("--ffmpeg", default=None,
        help="ffmpeg 실행 파일 전체 경로")
    parser.add_argument("--ffprobe", default=None,
        help="ffprobe 실행 파일 전체 경로")
    parser.add_argument("--dry-run", action="store_true",
        help="실제 추출 없이 계획만 출력")
    parser.add_argument("--no-meta", action="store_true",
        help="gs_metadata.json 저장 안 함")

    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.video).exists():
        print(f"오류: 파일을 찾을 수 없습니다 → {args.video}")
        sys.exit(1)

    # 출력 폴더
    if args.out:
        out_dir = Path(args.out)
    else:
        stem = Path(args.video).stem
        out_dir = Path(args.video).parent / f"{stem}_GS_Frames"

    EXEC_OVERRIDES["ffmpeg"] = args.ffmpeg
    EXEC_OVERRIDES["ffprobe"] = args.ffprobe

    try:
        plan = build_plan(args)
    except Exception as e:
        print(f"\n오류: {e}")
        sys.exit(1)

    try:
        extract_images(plan, out_dir, dry_run=args.dry_run)
    except FileNotFoundError as e:
        print(f"\n오류: {e}")
        sys.exit(1)

    if not args.no_meta and not args.dry_run:
        export_metadata(plan, out_dir)

    # 다음 단계 안내
    if not args.dry_run:
        print("다음 단계 (COLMAP → 가우시안 스플래팅):")
        print(f"  1. COLMAP feature_extractor --image_path {out_dir}")
        print(f"  2. COLMAP exhaustive_matcher  (또는 sequential_matcher)")
        print(f"  3. COLMAP mapper")
        print(f"  4. 3DGS / Nerfstudio 등으로 학습\n")


if __name__ == "__main__":
    main()