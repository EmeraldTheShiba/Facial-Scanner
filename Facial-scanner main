import os
from datetime import datetime

import cv2
import numpy as np
import psycopg
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from supabase import Client, create_client


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

ENROLL_DIR = "enroll"
UNKNOWN_DIR = "unknown"

THRESHOLD = 0.72
STREAK_NEEDED = 3
CAM_INDEX = 0
UNKNOWN_BUCKET = "unknown-faces"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY not found in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def init_dirs() -> None:
    os.makedirs(ENROLL_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_DIR, exist_ok=True)


def get_db_connection():
    return psycopg.connect(DATABASE_URL)


def load_gallery(app: FaceAnalysis):
    gallery = []

    print("\nLoading enrolled student images...\n")

    for name in os.listdir(ENROLL_DIR):
        person_dir = os.path.join(ENROLL_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        print(f"Checking folder: {name}")

        for img_name in os.listdir(person_dir):
            path = os.path.join(person_dir, img_name)
            img = cv2.imread(path)

            if img is None:
                print(f"  Skipped {img_name}: image could not be read")
                continue

            faces = app.get(img)
            if len(faces) != 1:
                print(f"  Skipped {img_name}: detected {len(faces)} faces")
                continue

            embeddings.append(l2norm(faces[0].embedding))
            print(f"  Loaded {img_name}")

        if embeddings:
            mean_embedding = l2norm(np.mean(embeddings, axis=0))
            gallery.append((name, mean_embedding, person_dir))
            print(f"  Added identity: {name}\n")
        else:
            print(f"  No valid images found for {name}\n")

    return gallery


def sync_students_to_db(gallery) -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for name, _, person_dir in gallery:
                cur.execute(
                    """
                    INSERT INTO students (name, image_path)
                    VALUES (%s, %s)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    (name, person_dir),
                )
        conn.commit()


def best_match(embedding: np.ndarray, gallery):
    best_name = None
    best_score = -1.0

    for name, ref_embedding, _ in gallery:
        score = cosine_sim(embedding, ref_embedding)
        if score > best_score:
            best_name = name
            best_score = score

    return best_name, best_score


def attendance_already_logged_today(name: str) -> bool:
    today = datetime.now().date()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM attendance
                WHERE name = %s
                  AND date = %s
                  AND status = 'Present'
                LIMIT 1
                """,
                (name, today),
            )
            row = cur.fetchone()

    return row is not None


def log_attendance(name: str, status: str, confidence=None) -> None:
    now = datetime.now()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance (name, date, time, status, confidence)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    name,
                    now.date(),
                    now.time().replace(microsecond=0),
                    status,
                    confidence,
                ),
            )
        conn.commit()

    print(
        f"LOGGED -> {name} | {now.date()} | {now.strftime('%H:%M:%S')} | "
        f"{status} | confidence={confidence}"
    )


def save_unknown_face(frame) -> str:
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
                {"content-type": "image/jpeg"},
            )

        stored_path = storage_path
        print(f"UNKNOWN FACE UPLOADED -> {storage_path}")

    except Exception as e:
        stored_path = local_path
        print(f"UPLOAD FAILED, SAVED LOCALLY -> {local_path}")
        print(f"UPLOAD ERROR -> {e}")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO unknown_faces (date, time, image_path, note)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    now.date(),
                    now.time().replace(microsecond=0),
                    stored_path,
                    "Unrecognized face during attendance session",
                ),
            )
        conn.commit()

    return stored_path


def main() -> None:
    init_dirs()

    print("Starting Automated Attendance Tracking System...")
    print("Loading facial recognition model...")

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    gallery = load_gallery(app)
    print(f"Loaded {len(gallery)} enrolled student identities.")

    if not gallery:
        print("No enrolled identities found. Add photos under enroll/<StudentName>/")
        return

    sync_students_to_db(gallery)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try CAM_INDEX = 1 or 2.")

    streak = 0
    last_name = None
    unknown_saved = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = datetime.now()
        date_text = now.strftime("%Y-%m-%d")
        time_text = now.strftime("%H:%M:%S")

        faces = app.get(frame)

        title_text = "Attendance System Ready"
        result_text = "Waiting for student..."
        detail_text = "No face detected"
        color = (255, 255, 255)

        if len(faces) == 1:
            embedding = l2norm(faces[0].embedding)
            name, score = best_match(embedding, gallery)
            detail_text = f"Best match: {name} | confidence={score:.3f}"

            if score >= THRESHOLD:
                color = (0, 255, 0)
                streak = streak + 1 if last_name == name else 1
                last_name = name
                unknown_saved = False

                if streak >= STREAK_NEEDED:
                    if not attendance_already_logged_today(last_name):
                        log_attendance(last_name, "Present", score)
                        result_text = f"Attendance Recorded: {last_name}"
                    else:
                        result_text = f"Already Marked Today: {last_name}"
                else:
                    result_text = f"Recognizing: {name}"
            else:
                streak = 0
                last_name = None
                color = (0, 0, 255)
                result_text = "Unknown Student"

                if not unknown_saved:
                    save_unknown_face(frame)
                    log_attendance("Unknown", "Unrecognized", score)
                    unknown_saved = True

        elif len(faces) > 1:
            streak = 0
            last_name = None
            color = (0, 165, 255)
            detail_text = "Multiple faces detected"
            result_text = "Please show one face only"

        else:
            streak = 0
            last_name = None

        cv2.putText(
            frame,
            "Automated Attendance Tracking System",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Date: {date_text}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Time: {time_text}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            title_text,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        cv2.putText(
            frame,
            result_text,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        cv2.putText(
            frame,
            detail_text,
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        cv2.putText(
            frame,
            "Press Q or ESC to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Automated Attendance Tracking System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
