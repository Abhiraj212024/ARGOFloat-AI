import duckdb
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_DB_PATH = _PROJECT_ROOT / "DB_files" / "data.duckdb"

_connection = None

def get_db_connection():
    global _connection
    if _connection is None:
        _connection = duckdb.connect(
            str(_DB_PATH),
            read_only=True
        )
    return _connection
