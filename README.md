# 360to3DGS

## 개요

`360to3DGS`는 Insta360 360° 영상에서 멀티뷰 이미지를 추출하고, 별도로 사람(person)을 자동으로 감지하여 이미지 파일을 정리하는 도구들을 포함한 프로젝트입니다.

이 저장소에는 다음 주요 스크립트가 포함되어 있습니다.

- `360to3DGS.py`: Insta360 360도 영상에서 멀티뷰 이미지를 추출하기 위한 Python 스크립트입니다.
- `erasehuman.py`: YOLOv8 모델을 사용해 이미지 내 사람을 감지하고, 검출된 이미지를 자동으로 삭제합니다.
- `BUILD_GUIDE.md`: exe 빌드와 배포 방법을 안내하는 문서입니다.
- `GETTING_STARTED.md`: 초기 설정 및 실행 방법을 설명하는 입문 가이드입니다.

## 주요 기능

### 360to3DGS.py

- `ffmpeg`의 `v360` 필터를 활용하여 360° 영상을 평면으로 투영하고 분할합니다.
- Pitch/Yaw 격자 방식으로 여러 뷰를 생성하여 3D 재구성 또는 멀티뷰 워크플로우에 활용할 수 있습니다.
- `outdoor`, `indoor`, `landscape`, `object`, `fastmove` 등 여러 프리셋을 통해 추출 설정을 조정할 수 있습니다.

### erasehuman.py

- `ultralytics`의 YOLOv8 모델(`yolov8n.pt`)을 사용합니다.
- 지정된 폴더의 이미지에서 클래스 번호 `0`(person)이 감지되면 해당 이미지를 삭제합니다.
- 지원 확장자: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

## 요구 사항

- Python 3.8 이상
- `ffmpeg`, `ffprobe` 설치 및 시스템 PATH에 추가 (360도 영상 추출용)
- `ultralytics` 패키지 설치 (사람 감지용)
- `yolov8n.pt` 모델 파일 (이미 저장소에 포함되어 있음)

## 설치 방법

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt`가 없는 경우, 필요한 패키지는 `flask`, `pyinstaller`, `ultralytics` 등입니다.

## 실행 방법

### 1. 360도 영상 추출

```bash
python 360to3DGS.py <video.mp4>
```

옵션과 자세한 사용법은 스크립트 내부 주석 및 출력 도움말을 참고하세요.

### 2. 사람 감지 이미지 삭제

`erasehuman.py` 파일에서 `target_folder` 값을 처리할 폴더 경로로 변경한 뒤 실행합니다.

```bash
python erasehuman.py
```

## 참고 문서

- `BUILD_GUIDE.md`: exe 빌드 및 배포 관련 가이드
- `GETTING_STARTED.md`: 프로젝트 초기 설정 및 실행 가이드

## 주의 사항

- `erasehuman.py`는 감지된 이미지를 즉시 삭제합니다. 실행 전 반드시 백업을 권장합니다.
- `360to3DGS.py` 실행에는 `ffmpeg`가 필수입니다.

## 라이선스

필요에 따라 라이선스 항목을 추가하세요.
