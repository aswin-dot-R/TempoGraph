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

CREATE TABLE IF NOT EXISTS audio_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    no_speech_prob REAL,
    avg_logprob REAL
);

CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_idx);
CREATE INDEX IF NOT EXISTS idx_audio_start ON audio_segments(start_ms);

CREATE TABLE IF NOT EXISTS run_stages (
    stage_name TEXT PRIMARY KEY,
    finished_at TEXT NOT NULL,
    elapsed_s REAL,
    n_units INTEGER
);

CREATE TABLE IF NOT EXISTS ethogram_labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_idx INTEGER NOT NULL,
    behavior TEXT NOT NULL,
    confidence REAL,
    note TEXT,
    profile_name TEXT NOT NULL DEFAULT 'default',
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);

CREATE INDEX IF NOT EXISTS idx_ethogram_frame ON ethogram_labels(frame_idx);
CREATE INDEX IF NOT EXISTS idx_ethogram_profile ON ethogram_labels(profile_name);

CREATE TABLE IF NOT EXISTS run_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
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

    def count_detections(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        )

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

    def insert_audio_segment(
        self,
        start_ms: int,
        end_ms: int,
        text: str,
        no_speech_prob: Optional[float] = None,
        avg_logprob: Optional[float] = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO audio_segments "
            "(start_ms, end_ms, text, no_speech_prob, avg_logprob) "
            "VALUES (?, ?, ?, ?, ?)",
            (start_ms, end_ms, text, no_speech_prob, avg_logprob),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_audio_segments(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM audio_segments ORDER BY start_ms ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def count_audio_segments(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM audio_segments").fetchone()[0]
        )

    # ── stage tracking (crash-resume) ──────────────────────────────

    def mark_stage_complete(
        self, stage_name: str, elapsed_s: float = 0.0, n_units: int = 0,
    ) -> None:
        """Record that a pipeline stage finished successfully."""
        from datetime import datetime
        self._conn.execute(
            "INSERT OR REPLACE INTO run_stages "
            "(stage_name, finished_at, elapsed_s, n_units) VALUES (?, ?, ?, ?)",
            (stage_name, datetime.utcnow().isoformat(), elapsed_s, n_units),
        )
        self._conn.commit()

    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage was previously completed."""
        cur = self._conn.execute(
            "SELECT stage_name FROM run_stages WHERE stage_name = ?",
            (stage_name,),
        )
        return cur.fetchone() is not None

    def clear_stages(self) -> None:
        """Clear all stage completion records (fresh run)."""
        self._conn.execute("DELETE FROM run_stages")
        self._conn.commit()

    # ── run metadata ─────────────────────────────────────────────────

    def get_meta(self, key: str) -> Optional[str]:
        """Read a run_meta value, or None."""
        cur = self._conn.execute(
            "SELECT value FROM run_meta WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Store a run_meta key-value pair."""
        self._conn.execute(
            "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def delete_meta(self, key: str) -> None:
        """Remove a run_meta key."""
        self._conn.execute("DELETE FROM run_meta WHERE key = ?", (key,))
        self._conn.commit()

    # ── ethogram labels ────────────────────────────────────────────

    def insert_ethogram_label(
        self,
        frame_idx: int,
        behavior: str,
        confidence: Optional[float] = None,
        note: Optional[str] = None,
        profile_name: str = "default",
    ) -> int:
        """Insert a per-frame ethogram behavior label."""
        cur = self._conn.execute(
            "INSERT INTO ethogram_labels "
            "(frame_idx, behavior, confidence, note, profile_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (frame_idx, behavior, confidence, note, profile_name),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_ethogram_labels(self, profile_name: str = "default") -> list:
        """Get all ethogram labels for a profile, ordered by frame index."""
        rows = self._conn.execute(
            "SELECT * FROM ethogram_labels WHERE profile_name = ? "
            "ORDER BY frame_idx ASC",
            (profile_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_ethogram_profile(self, profile_name: str = "default") -> int:
        """Delete all labels for a given profile. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM ethogram_labels WHERE profile_name = ?",
            (profile_name,),
        )
        self._conn.commit()
        return cur.rowcount

    def list_ethogram_profiles(self) -> list:
        """List distinct ethogram profile names."""
        rows = self._conn.execute(
            "SELECT DISTINCT profile_name FROM ethogram_labels ORDER BY profile_name"
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
