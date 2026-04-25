"""SQLite-backed storage for TempoGraph v2 pipeline."""

import sqlite3
from pathlib import Path
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_idx INTEGER PRIMARY KEY,
    timestamp_ms INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    is_keyframe INTEGER NOT NULL,
    delta_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_idx INTEGER NOT NULL,
    track_id INTEGER,
    class_name TEXT NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    confidence REAL NOT NULL,
    mean_depth REAL,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);

CREATE TABLE IF NOT EXISTS depth_frames (
    frame_idx INTEGER PRIMARY KEY,
    depth_npy_path TEXT NOT NULL,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);

CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_idx);
"""


class TempoGraphDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def has_table(self, name: str) -> bool:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        return cur.fetchone() is not None

    def insert_frame(
        self,
        frame_idx: int,
        timestamp_ms: int,
        image_path: str,
        is_keyframe: bool,
        delta_score: float,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO frames (frame_idx, timestamp_ms, image_path, is_keyframe, delta_score) "
            "VALUES (?, ?, ?, ?, ?)",
            (frame_idx, timestamp_ms, image_path, 1 if is_keyframe else 0, delta_score),
        )
        self._conn.commit()

    def get_frame(self, frame_idx: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM frames WHERE frame_idx = ?", (frame_idx,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_frame_indices(self) -> list:
        rows = self._conn.execute(
            "SELECT frame_idx FROM frames ORDER BY frame_idx ASC"
        ).fetchall()
        return [r["frame_idx"] for r in rows]

    def insert_detection(
        self,
        frame_idx: int,
        track_id: Optional[int],
        class_name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        mean_depth: Optional[float] = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO detections "
            "(frame_idx, track_id, class_name, x1, y1, x2, y2, confidence, mean_depth) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (frame_idx, track_id, class_name, x1, y1, x2, y2, confidence, mean_depth),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_detections_for_frame(self, frame_idx: int) -> list:
        rows = self._conn.execute(
            "SELECT * FROM detections WHERE frame_idx = ? ORDER BY detection_id ASC",
            (frame_idx,),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_detection_mean_depth(self, detection_id: int, mean_depth: float) -> None:
        self._conn.execute(
            "UPDATE detections SET mean_depth = ? WHERE detection_id = ?",
            (mean_depth, detection_id),
        )
        self._conn.commit()

    def insert_depth_frame(self, frame_idx: int, depth_npy_path: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO depth_frames (frame_idx, depth_npy_path) VALUES (?, ?)",
            (frame_idx, depth_npy_path),
        )
        self._conn.commit()

    def get_depth_path(self, frame_idx: int) -> Optional[str]:
        row = self._conn.execute(
            "SELECT depth_npy_path FROM depth_frames WHERE frame_idx = ?",
            (frame_idx,),
        ).fetchone()
        return row["depth_npy_path"] if row else None

    def close(self) -> None:
        self._conn.close()
