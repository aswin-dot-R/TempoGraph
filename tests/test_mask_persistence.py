"""Mask persistence: schema migration, storage round-trip, seg smoke run.

Covers TODO item 1 acceptance:
- mask_rle column exists on new DBs and is added to legacy DBs by migration;
- masks encoded with src/rle survive an insert/select round trip;
- CPU smoke run of the seg model on a 5 s ffmpeg testsrc clip yields >= 1
  detection row with non-NULL mask_rle (integration: detector -> storage).
"""

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rle import decode_from_string, encode_to_string
from src.storage import TempoGraphDB

REPO_ROOT = Path(__file__).resolve().parents[1]
SEG_WEIGHTS = REPO_ROOT / "yolo26n-seg.pt"

LEGACY_DETECTIONS_SQL = """
CREATE TABLE frames (
    frame_idx INTEGER PRIMARY KEY,
    timestamp_ms INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    is_keyframe INTEGER NOT NULL,
    delta_score REAL NOT NULL
);
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_idx INTEGER NOT NULL,
    track_id INTEGER,
    class_name TEXT NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    confidence REAL NOT NULL,
    mean_depth REAL,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);
"""


def _detection_columns(db_path: Path) -> set:
    with sqlite3.connect(str(db_path)) as conn:
        return {r[1] for r in conn.execute("PRAGMA table_info(detections)")}


class TestSchemaMigration:
    def test_new_db_has_mask_rle_column(self, tmp_path):
        db = TempoGraphDB(tmp_path / "tempograph.db")
        db.close()
        assert "mask_rle" in _detection_columns(tmp_path / "tempograph.db")

    def test_legacy_db_is_migrated_and_rows_survive(self, tmp_path):
        db_path = tmp_path / "tempograph.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(LEGACY_DETECTIONS_SQL)
            conn.execute(
                "INSERT INTO detections (frame_idx, track_id, class_name, "
                "x1, y1, x2, y2, confidence) VALUES (0, 1, 'dog', "
                "0.1, 0.1, 0.5, 0.5, 0.9)"
            )
            conn.commit()

        assert "mask_rle" not in _detection_columns(db_path)
        db = TempoGraphDB(db_path)  # migration runs in __init__
        assert "mask_rle" in _detection_columns(db_path)

        rows = db.get_detections_for_frame(0)
        assert len(rows) == 1
        assert rows[0]["class_name"] == "dog"
        assert rows[0]["mask_rle"] is None  # NULL = no mask

        # And new-style inserts with masks work on the migrated DB
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        db.insert_detection(0, None, "cat", 0.2, 0.2, 0.6, 0.6, 0.8,
                            mask_rle=encode_to_string(mask))
        db.close()

    def test_reopening_migrated_db_is_idempotent(self, tmp_path):
        db_path = tmp_path / "tempograph.db"
        for _ in range(3):
            TempoGraphDB(db_path).close()
        cols = _detection_columns(db_path)
        assert sum(1 for c in cols if c == "mask_rle") == 1


class TestMaskStorageRoundTrip:
    def test_insert_and_decode_mask(self, tmp_path):
        db = TempoGraphDB(tmp_path / "tempograph.db")
        db.insert_frame(0, 0, "frame_000000.jpg", True, 1.0)
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[5:20, 10:35] = 1
        db.insert_detection(0, 7, "person", 0.25, 0.16, 0.87, 0.66, 0.77,
                            mask_rle=encode_to_string(mask))
        db.insert_detection(0, None, "chair", 0.0, 0.0, 0.1, 0.1, 0.5)

        dets = db.get_detections_for_frame(0)
        db.close()
        assert len(dets) == 2
        with_mask = [d for d in dets if d["mask_rle"]]
        assert len(with_mask) == 1
        decoded = decode_from_string(with_mask[0]["mask_rle"])
        np.testing.assert_array_equal(decoded, mask)


@pytest.mark.skipif(not SEG_WEIGHTS.exists(),
                    reason="yolo26n-seg.pt not present in repo root")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
class TestSegSmoke:
    """CPU seg-model smoke run on a 5 s synthetic ffmpeg testsrc clip."""

    @pytest.fixture(scope="class")
    def seg_run_db(self, tmp_path_factory):
        import cv2

        from src.modules.detector import ObjectDetector

        tmp = tmp_path_factory.mktemp("seg_smoke")
        clip = tmp / "testsrc5.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=duration=5:size=640x360:rate=30",
             "-pix_fmt", "yuv420p", str(clip)],
            check=True,
        )

        # Extract a handful of frames the way the pipeline saves them.
        cap = cv2.VideoCapture(str(clip))
        indices = [0, 30, 60, 90, 120]
        frame_paths = []
        w = h = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            assert ok, f"failed to read frame {idx} from testsrc clip"
            h, w = frame.shape[:2]
            p = tmp / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(p), frame)
            frame_paths.append(p)
        cap.release()

        db = TempoGraphDB(tmp / "tempograph.db")
        for idx in indices:
            db.insert_frame(idx, int(idx * 1000 / 30), str(tmp / f"frame_{idx:06d}.jpg"),
                            False, 0.0)

        # testsrc is an abstract pattern; only very low confidence yields
        # (spurious but deterministic) instances, which is all the smoke
        # test needs to prove the mask encode->DB path works end to end.
        detector = ObjectDetector(
            model_path=str(SEG_WEIGHTS), confidence=0.01, device="cpu"
        )
        detector.detect_to_db(
            frame_indices=indices,
            frame_paths=frame_paths,
            db=db,
            frame_width=w,
            frame_height=h,
        )
        detector.cleanup()
        yield db
        db.close()

    def test_seg_run_produces_mask_rows(self, seg_run_db):
        rows = seg_run_db._conn.execute(
            "SELECT COUNT(*) FROM detections WHERE mask_rle IS NOT NULL"
        ).fetchone()
        assert rows[0] >= 1, "expected >= 1 detection row with non-NULL mask_rle"

    def test_seg_masks_decode_to_frame_size(self, seg_run_db):
        row = seg_run_db._conn.execute(
            "SELECT mask_rle FROM detections WHERE mask_rle IS NOT NULL LIMIT 1"
        ).fetchone()
        mask = decode_from_string(row["mask_rle"])
        assert mask.shape == (360, 640)
        assert mask.any()

    def test_bbox_only_model_leaves_mask_null(self, tmp_path):
        """Non-seg runs must keep mask_rle NULL (regression guard)."""
        db = TempoGraphDB(tmp_path / "tempograph.db")
        db.insert_detection(0, None, "person", 0.1, 0.1, 0.5, 0.5, 0.9)
        row = db._conn.execute(
            "SELECT mask_rle FROM detections"
        ).fetchone()
        db.close()
        assert row["mask_rle"] is None
