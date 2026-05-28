import os
from pathlib import Path
from ultralytics import YOLO

# erasehuman.py
# 기능: 지정한 폴더의 이미지들을 YOLOv8 모델로 검사하여
#       사람(person, 클래스 0)이 감지된 이미지 파일을 삭제합니다.
#
# 사용법:
#  - `delete_person_images(path)` 함수를 호출하거나, 아래 `target_folder`에
#    대상 폴더 경로를 지정한 뒤 스크립트를 실행하세요.
#
# 요구사항 및 권장 사항:
#  - Python 3.8 이상
#  - ultralytics 패키지 설치 (권장: 프로젝트 가상환경 사용, 예: .venv)
#  - YOLOv8 모델 파일 (예: yolov8n.pt)이 접근 가능한 위치에 있어야 합니다.
#    필요하면 모델 경로를 변경해서 다른 모델을 사용하세요.
#
# 동작 요약:
#  - 지원 확장자: .jpg, .jpeg, .png, .webp, .bmp
#  - 각 이미지를 모델로 분석하여 클래스 번호 0(person)이 검출되면 파일을 삭제합니다.
#
# 주의:
#  - 삭제는 영구적일 수 있으니 실행 전 백업을 권장합니다.
#  - 안전을 위해 삭제 대신 별도 폴더로 이동하도록 코드를 수정할 수 있습니다.


def delete_person_images(folder_path):
    # 1. 고성능 YOLOv8 모델 로드 (가장 가벼운 nano 버전 사용)
    model = YOLO("yolov8n.pt")
    
    # 지원하는 이미지 확장자 정의
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # 폴더 안의 모든 파일 탐색
    folder = Path(folder_path)
    if not folder.exists():
        print(f"오류: '{folder_path}' 폴더를 찾을 수 없습니다.")
        return
        
    image_paths = [p for p in folder.iterdir() if p.suffix.lower() in valid_extensions]
    
    if not image_paths:
        print("처리할 이미지 파일이 없습니다.")
        return
        
    print(f"총 {len(image_paths)}개의 이미지를 분석합니다...\n")
    deleted_count = 0

    for img_path in image_paths:
        try:
            # 2. 이미지에서 객체 감지 수행 (verbose=False로 로그 간소화)
            results = model(str(img_path), verbose=False)
            
            is_person_detected = False
            
            # 감지된 객체들 확인
            for result in results:
                for box in result.boxes:
                    # YOLO 데이터셋에서 클래스 번호 0은 'person'(사람)을 의미합니다.
                    if int(box.cls[0]) == 0:
                        is_person_detected = True
                        break
                if is_person_detected:
                     Aurora_break = True
            
            # 3. 사람이 감지되었다면 파일 삭제
            if is_person_detected:
                os.remove(img_path)
                print(f"[삭제 완료] 사람이 포함됨: {img_path.name}")
                deleted_count += 1
            else:
                print(f"[유지] 사람 없음: {img_path.name}")
                
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({img_path.name}): {e}")

    print("\n" + "="*40)
    print(f"작업 완료! 총 {deleted_count}개의 사람이 나온 사진을 삭제했습니다.")
    print("="*40)

# --- 실행 코드 ---
if __name__ == "__main__":
    # 여기에 사진이 들어있는 폴더 경로를 입력하세요.
    # 예: r"C:\Users\Username\Pictures\MyFolder" (앞에 r을 붙이면 좋습니다)
    target_folder = r"행당_폴더_경로를_여기에_입력하세요" 
    
    delete_person_images(target_folder)