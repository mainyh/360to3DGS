#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
erasehuman_gui.py
GUI 기반 사람 이미지 삭제 도구 (YOLOv8)

기능:
 - 폴더 선택 다이얼로그
 - 실시간 진행 상황 표시
 - 완료 후 결과 요약

요구사항:
 - Python 3.8 이상
 - ultralytics 패키지
"""

import os
import sys
import threading
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, ttk, scrolledtext
from ultralytics import YOLO


class ErazeHumanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("사람 이미지 삭제 도구 (Person Image Remover)")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        self.model = None
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.is_running = False
        
        # UI 구성
        self._build_ui()
        
    def _build_ui(self):
        """UI 구성"""
        # 1. 폴더 선택 프레임
        frame_folder = ttk.LabelFrame(self.root, text="폴더 선택", padding=10)
        frame_folder.pack(fill="x", padx=10, pady=10)
        
        self.label_path = ttk.Label(frame_folder, text="선택된 폴더: 없음", foreground="gray")
        self.label_path.pack(fill="x", pady=5)
        
        self.btn_select = ttk.Button(frame_folder, text="폴더 선택...", command=self._select_folder)
        self.btn_select.pack(fill="x", pady=5)
        
        # 2. 진행 상황 프레임
        frame_progress = ttk.LabelFrame(self.root, text="진행 상황", padding=10)
        frame_progress.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.progress_bar = ttk.Progressbar(frame_progress, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=5)
        
        self.label_status = ttk.Label(frame_progress, text="준비 완료", foreground="blue")
        self.label_status.pack(fill="x", pady=5)
        
        # 로그 영역
        self.log_text = scrolledtext.ScrolledText(frame_progress, height=10, width=70, state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=5)
        
        # 3. 버튼 프레임
        frame_buttons = ttk.Frame(self.root)
        frame_buttons.pack(fill="x", padx=10, pady=10)
        
        self.btn_run = ttk.Button(frame_buttons, text="실행 시작", command=self._run_deletion, state="disabled")
        self.btn_run.pack(side="left", padx=5)
        
        self.btn_stop = ttk.Button(frame_buttons, text="중지", command=self._stop_execution, state="disabled")
        self.btn_stop.pack(side="left", padx=5)
        
        ttk.Button(frame_buttons, text="종료", command=self.root.quit).pack(side="right", padx=5)
        
    def _select_folder(self):
        """폴더 선택 다이얼로그"""
        folder = filedialog.askdirectory(title="처리할 폴더 선택")
        if folder:
            self.selected_folder = folder
            self.label_path.config(text=f"선택된 폴더: {folder}", foreground="black")
            self.btn_run.config(state="normal")
            self._log("폴더 선택됨: " + folder)
            
    def _log(self, message):
        """로그에 메시지 추가"""
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update()
        
    def _run_deletion(self):
        """삭제 작업 실행 (스레드)"""
        if not hasattr(self, 'selected_folder'):
            messagebox.showwarning("경고", "폴더를 선택해주세요.")
            return
        
        self.is_running = True
        self.btn_run.config(state="disabled")
        self.btn_select.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.progress_bar.start()
        
        thread = threading.Thread(target=self._delete_person_images, daemon=True)
        thread.start()
        
    def _stop_execution(self):
        """실행 중지"""
        self.is_running = False
        self.btn_stop.config(state="disabled")
        self._log("실행이 중지되었습니다.")
        
    def _delete_person_images(self):
        """사람이 포함된 이미지 삭제"""
        try:
            self._log("YOLOv8 모델 로딩 중...")
            self.label_status.config(text="상태: 모델 로딩", foreground="orange")
            
            if self.model is None:
                self.model = YOLO("yolov8n.pt")
            self._log("✓ 모델 로딩 완료")
            
            folder = Path(self.selected_folder)
            if not folder.exists():
                self._log(f"❌ 오류: '{folder}' 폴더를 찾을 수 없습니다.")
                self.label_status.config(text="상태: 오류 발생", foreground="red")
                return
            
            image_paths = [p for p in folder.iterdir() if p.suffix.lower() in self.valid_extensions]
            if not image_paths:
                self._log("처리할 이미지 파일이 없습니다.")
                self.label_status.config(text="상태: 완료 (이미지 없음)", foreground="blue")
                return
            
            self._log(f"분석할 이미지: {len(image_paths)}개")
            self.label_status.config(text=f"상태: 분석 중... (0/{len(image_paths)})", foreground="orange")
            
            deleted_count = 0
            
            for idx, img_path in enumerate(image_paths):
                if not self.is_running:
                    self._log("사용자가 중지함")
                    break
                
                try:
                    self.label_status.config(text=f"상태: 분석 중... ({idx+1}/{len(image_paths)})")
                    self.root.update()
                    
                    # 이미지 분석
                    results = self.model(str(img_path), verbose=False)
                    
                    is_person_detected = False
                    for result in results:
                        for box in result.boxes:
                            if int(box.cls[0]) == 0:  # person 클래스
                                is_person_detected = True
                                break
                        if is_person_detected:
                            break
                    
                    # 삭제 또는 유지
                    if is_person_detected:
                        os.remove(img_path)
                        deleted_count += 1
                        msg = f"[삭제] {img_path.name}"
                        self._log(msg)
                    else:
                        msg = f"[유지] {img_path.name}"
                        self._log(msg)
                        
                except Exception as e:
                    self._log(f"[오류] {img_path.name}: {str(e)}")
            
            # 완료 요약
            self.progress_bar.stop()
            self.label_status.config(text="상태: 완료", foreground="green")
            
            summary = (
                f"\n{'='*50}\n"
                f"작업 완료!\n"
                f"총 이미지 수: {len(image_paths)}\n"
                f"삭제된 파일: {deleted_count}개\n"
                f"유지된 파일: {len(image_paths) - deleted_count}개\n"
                f"{'='*50}"
            )
            self._log(summary)
            
            messagebox.showinfo("완료", f"작업 완료!\n\n삭제됨: {deleted_count}개\n유지됨: {len(image_paths) - deleted_count}개")
            
        except Exception as e:
            self.progress_bar.stop()
            self.label_status.config(text="상태: 오류 발생", foreground="red")
            self._log(f"❌ 오류: {str(e)}")
            messagebox.showerror("오류", f"오류가 발생했습니다:\n{str(e)}")
        
        finally:
            self.btn_run.config(state="normal")
            self.btn_select.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.progress_bar.stop()


def main():
    root = Tk()
    gui = ErazeHumanGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
