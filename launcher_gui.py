import os
import sys
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog

ROOT = Path(__file__).parent


def get_python_executable():
    venv_python = ROOT / '.venv' / 'Scripts' / 'python.exe'
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_script(script_name):
    script_path = ROOT / script_name
    if not script_path.exists():
        messagebox.showerror('오류', f'파일이 존재하지 않습니다:\n{script_path}')
        return

    python = get_python_executable()
    args = [python, str(script_path)]

    def _start():
        try:
            # Windows: open in new console so the GUI stays responsive
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NEW_CONSOLE
                subprocess.Popen(args, creationflags=creationflags)
            else:
                subprocess.Popen(args)
        except Exception as e:
            messagebox.showerror('실행 오류', str(e))

    threading.Thread(target=_start, daemon=True).start()


def run_pyinstaller():
    python = get_python_executable()
    cmd = [python, '-m', 'PyInstaller', '--onefile', '--noconsole', '--name', '360to3DGS_UI', 'launcher_gui.py']

    def _start_build():
        try:
            if os.name == 'nt':
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd)
            messagebox.showinfo('빌드', 'PyInstaller 빌드는 새 콘솔에서 실행됩니다. 콘솔 출력을 확인하세요.')
        except Exception as e:
            messagebox.showerror('빌드 오류', str(e))

    threading.Thread(target=_start_build, daemon=True).start()


def open_folder():
    try:
        os.startfile(str(ROOT))
    except Exception:
        messagebox.showinfo('폴더 열기', f'경로: {ROOT}')


app = tk.Tk()
app.title('360to3DGS Launcher')
app.geometry('420x220')

lbl = tk.Label(app, text='360to3DGS 실행기', font=('Segoe UI', 14))
lbl.pack(pady=8)

py_label = tk.Label(app, text=f'Python: {get_python_executable()}', wraplength=380)
py_label.pack(pady=4)

frame = tk.Frame(app)
frame.pack(pady=8)

btn_web = tk.Button(frame, text='Run Web UI (insta360_gs_gui.py)', width=32, command=lambda: run_script('insta360_gs_gui.py'))
btn_web.grid(row=0, column=0, padx=6, pady=4)

btn_erase = tk.Button(frame, text='Run EraseHuman GUI (erasehuman_gui.py)', width=32, command=lambda: run_script('erasehuman_gui.py'))
btn_erase.grid(row=1, column=0, padx=6, pady=4)

btn_open = tk.Button(frame, text='Open Project Folder', width=32, command=open_folder)
btn_open.grid(row=2, column=0, padx=6, pady=4)

btn_build = tk.Button(app, text='Build EXE (PyInstaller)', fg='white', bg='#0078D7', command=run_pyinstaller)
btn_build.pack(pady=6)

note = tk.Label(app, text='빌드 전에 .venv 활성화 및 PyInstaller 설치가 필요합니다.', wraplength=380, fg='gray')
note.pack(side='bottom', pady=6)

if __name__ == '__main__':
    app.mainloop()
