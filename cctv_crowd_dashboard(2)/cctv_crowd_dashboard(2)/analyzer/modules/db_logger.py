import sqlite3
from pathlib import Path


class DBLogger:
    BATCH_SIZE = 30

    def __init__(self, db_path: Path, video_name: str):
        self.video_name = video_name
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._batch = []
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS crowd_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name    TEXT    NOT NULL,
                frame_index   INTEGER NOT NULL,
                in_count      INTEGER NOT NULL,
                out_count     INTEGER NOT NULL,
                roi_person_count INTEGER NOT NULL,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def insert(self, frame_index: int, in_count: int, out_count: int, roi_person_count: int):
        self._batch.append((self.video_name, frame_index, in_count, out_count, roi_person_count))
        if len(self._batch) >= self.BATCH_SIZE:
            self._flush()

    def _flush(self):
        if not self._batch:
            return
        self.conn.executemany(
            "INSERT INTO crowd_log (video_name, frame_index, in_count, out_count, roi_person_count) VALUES (?, ?, ?, ?, ?)",
            self._batch,
        )
        self.conn.commit()
        self._batch = []

    def close(self):
        self._flush()
        self.conn.close()
