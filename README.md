# 🚀 Automated Attendance Tracking System

An AI-powered facial recognition attendance system built with Python, InsightFace, OpenCV, and Supabase.

---

## 📌 Features

- 🎥 Real-time face recognition via webcam
- 🧠 AI-based identity matching (InsightFace)
- ☁️ Cloud database integration (Supabase PostgreSQL)
- 📸 Unknown face detection + cloud storage
- 🖥️ GUI launcher interface
- 📊 Attendance log viewer

---

## 🛠️ Tech Stack

- Python
- OpenCV
- InsightFace
- Supabase (PostgreSQL + Storage)
- Tkinter (GUI)

---

## 📂 Project Structure
face_auth/
├── main.py
├── database.py
├── launcher.py
├── report_viewer.py
├── .env
├── README.md
├── requirements.txt

🧠 How It Works
1.Loads enrolled student faces from enroll/
2.Generates face embeddings using InsightFace
3.Matches webcam input against stored embeddings
4.Logs attendance in Supabase database
5.Uploads unknown faces to cloud storage
📊 Future Improvements
1.Better recognition accuracy
2.Admin dashboard
3.Mobile integration
4.Multi-user analytics

<img width="1905" height="896" alt="Screenshot 2026-04-06 135541" src="https://github.com/user-attachments/assets/2f01d8f1-d840-4058-af35-ab0386359758" />


<img width="1904" height="1106" alt="image" src="https://github.com/user-attachments/assets/3d090ee7-c517-426a-9a42-211719947f00" />





