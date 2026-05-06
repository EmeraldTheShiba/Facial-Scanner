import os
from datetime import datetime

import cv2
import numpy as np
import psycopg
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from supabase import create_client


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

UNKNOWN_DIR = "unknown"
UNKNOWN_BUCKET = "unknown-faces"

THRESHOLD = 0.72
STREAK_NEEDED = 3
CAM_INDEX = 0

if not DATABASE_URL:
    raise ValueError("DATABASE_URL missing")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials missing")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================
# Utils
# =============================
def l2norm(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_sim(a, b):
    return float(np.dot(a, b))


def get_db():
    return psycopg.connect(DATABASE_URL)


def init_dirs():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)


# =============================
# Load from Supabase (IMPORTANT)
# =============================
def load_gallery_from_db():
    gallery = []

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.full_name, fe.embedding::text
                FROM face_embeddings fe
                JOIN students s ON s.id = fe.student_id
            """)
            rows = cur.fetchall()

    for name, embedding_text in rows:
        cleaned = embedding_text.strip("[]")
        values = np.fromstring(cleaned, sep=",", dtype=np.float32)

        if values.size == 0:
            continue

        gallery.append((name, l2norm(values)))

    print(f"Loaded {len(gallery)} students from Supabase.")
    return gallery


# =============================
# Matching
# =============================
def best_match(embedding, gallery):
    best_name = None
    best_score = -1.0

    for name, ref_embedding in gallery:
        score = cosine_sim(embedding, ref_embedding)
        if score > best_score:
            best_name = name
            best_score = score

    return best_name, best_score


# =============================
# Attendance
# =============================
def already_marked_today(name):
    today = datetime.now().date()

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM attendance
                WHERE name = %s AND date = %s AND status = 'Present'
                LIMIT 1
            """, (name, today))
            return cur.fetchone() is not None


def log_attendance(name, status, confidence):
    now = datetime.now()

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO attendance (name, date, time, status, confidence)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                name,
                now.date(),
                now.time().replace(microsecond=0),
                status,
                confidence
            ))
        conn.commit()

    print(f"LOGGED -> {name} | {status} | {confidence:.3f}")


# =============================
# Unknown Faces
# =============================
def save_unknown(frame):
    now = datetime.now()
    filename = f"unknown_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    local_path = os.path.join(UNKNOWN_DIR, filename)
    storage_path = f"unknown/{filename}"

    cv2.imwrite(local_path, frame)

    try:
        with open(local_path, "rb") as f:
            supabase.storage.from_(UNKNOWN_BUCKET).upload(
                storage_path,
                f,
                {"content-type": "image/jpeg", "x-upsert": "true"}
            )

        print(f"UPLOADED UNKNOWN -> {storage_path}")
        path_used = storage_path

    except Exception as e:
        print("UPLOAD FAILED:", e)
        path_used = local_path

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO unknown_faces (date, time, image_path, note)
                VALUES (%s, %s, %s, %s)
            """, (
                now.date(),
                now.time().replace(microsecond=0),
                path_used,
                "Unknown face"
            ))
        conn.commit()


# =============================
# MAIN
# =============================
# =============================
# MAIN
# =============================
def main():
    init_dirs()

    print("Starting system...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320, 320))

    gallery = load_gallery_from_db()

    if not gallery:
        print("No students found in Supabase")
        return

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    streak = 0
    last_name = None
    unknown_saved = False
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 2 != 0:
            continue

        faces = app.get(frame)

        result_text = "Waiting..."
        color = (255, 255, 255)

        if len(faces) == 1:
            embedding = l2norm(faces[0].embedding)
            name, score = best_match(embedding, gallery)

            if score >= THRESHOLD:
                streak = streak + 1 if name == last_name else 1
                last_name = name
                unknown_saved = False
                color = (0, 255, 0)

                if streak >= STREAK_NEEDED:
                    if not already_marked_today(name):
                        log_attendance(name, "Present", score)
                        result_text = f"Attendance Recorded: {name}"
                    else:
                        result_text = f"Already Marked: {name}"
                else:
                    result_text = f"Recognizing: {name}"

            else:
                streak = 0
                last_name = None
                color = (0, 0, 255)
                result_text = "Unknown"

                if not unknown_saved:
                    save_unknown(frame)
                    log_attendance("Unknown", "Unrecognized", score)
                    unknown_saved = True

        cv2.putText(
            frame,
            result_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
