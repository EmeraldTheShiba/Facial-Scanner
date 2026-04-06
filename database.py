import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")


def init_db():
    print("Connecting to database...")
    print(DATABASE_URL.split("@")[-1])  # prints host/db only

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    image_path TEXT NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    status TEXT NOT NULL,
                    confidence DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS unknown_faces (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    image_path TEXT,
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        conn.commit()

    print("Remote attendance database initialized successfully.")


if __name__ == "__main__":
    init_db()