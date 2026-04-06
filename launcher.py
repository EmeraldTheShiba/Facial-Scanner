import tkinter as tk
import subprocess
import sys


def start_attendance():
    subprocess.Popen([sys.executable, "main.py"])


def view_logs():
    subprocess.Popen([sys.executable, "report_viewer.py"])


def exit_app():
    root.destroy()


root = tk.Tk()
root.title("Attendance System")
root.geometry("500x300")
root.configure(bg="#f4f4f4")

title = tk.Label(
    root,
    text="Automated Attendance System",
    font=("Arial", 18, "bold"),
    bg="#f4f4f4"
)
title.pack(pady=20)

start_btn = tk.Button(
    root,
    text="Start Attendance",
    font=("Arial", 12),
    width=25,
    height=2,
    command=start_attendance,
    bg="#4CAF50",
    fg="white"
)
start_btn.pack(pady=10)

log_btn = tk.Button(
    root,
    text="View Attendance Logs",
    font=("Arial", 12),
    width=25,
    height=2,
    command=view_logs,
    bg="#2196F3",
    fg="white"
)
log_btn.pack(pady=10)

exit_btn = tk.Button(
    root,
    text="Exit",
    font=("Arial", 12),
    width=25,
    height=2,
    command=exit_app,
    bg="#f44336",
    fg="white"
)
exit_btn.pack(pady=10)

root.mainloop()