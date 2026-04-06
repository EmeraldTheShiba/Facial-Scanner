import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")


def main():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT name, date, time, status, confidence
                FROM attendance
                ORDER BY id DESC
            """)
            rows = cur.fetchall()

    print("\nATTENDANCE LOGS\n")
    print("-" * 90)

    if not rows:
        print("No attendance records found.")
    else:
        for row in rows:
            name, date_val, time_val, status, confidence = row
            print(
                f"Name: {name} | Date: {date_val} | Time: {time_val} | "
                f"Status: {status} | Confidence: {confidence}"
            )

    print("-" * 90)


if __name__ == "__main__":
    main()
