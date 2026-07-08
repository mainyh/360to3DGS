Windows용 실행기 및 빌드 안내

빠른 실행

1. 가상환경 활성화 (PowerShell):

```
& d:\GIT\360to3DGS\.venv\Scripts\Activate.ps1
```

2. 실행기 시작:

```
& .venv\Scripts\python.exe launcher_gui.py
```

주요 버튼

- Run Web UI: `insta360_gs_gui.py`를 새 콘솔에서 실행
- Run EraseHuman GUI: `erasehuman_gui.py`를 새 콘솔에서 실행
- Open Project Folder: 프로젝트 폴더를 엽니다
- Build EXE (PyInstaller): `launcher_gui.py`를 PyInstaller로 빌드합니다

EXE 빌드

1. 가상환경에서 PyInstaller 설치:

```
& .venv\Scripts\python.exe -m pip install pyinstaller
```

2. 빌드 명령 (프로젝트 루트에서 실행):

```
& .venv\Scripts\python.exe -m PyInstaller --onefile --noconsole --name 360to3DGS_UI launcher_gui.py
```

3. 결과: `dist\\360to3DGS_UI.exe` 생성

주의

- standalone EXE로 만들 때, 런처가 호출하는 다른 스크립트(`insta360_gs_gui.py`, `erasehuman_gui.py`)가 함께 필요하면 별도 패키징이 필요합니다.
- 빌드 전 `.venv`가 활성화된 상태와 필요한 패키지 설치 여부를 확인하세요.
