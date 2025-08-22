# tools/init_db.py
from storage import init_db, DB_PATH

if __name__ == "__main__":
    init_db()
    print(f"SQLite initialized at {DB_PATH}")
