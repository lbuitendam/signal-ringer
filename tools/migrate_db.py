# tools/migrate_db.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import storage.py from root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import init_db  # now resolvable from project root


def main() -> None:
    init_db()
    db_path = ROOT / "data" / "signal_ringer.db"
    print(f"[OK] Database initialized at: {db_path}")


if __name__ == "__main__":
    main()
