import tempfile
from pathlib import Path

from src.storage import TempoGraphDB


def test_db_creates_schema():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        assert db.has_table("frames")
        assert db.has_table("detections")
        assert db.has_table("depth_frames")
        db.close()


def test_insert_and_get_frame():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        db.insert_frame(
            frame_idx=10,
            timestamp_ms=333,
            image_path="/tmp/f.jpg",
            is_keyframe=True,
            delta_score=8.4,
        )
        row = db.get_frame(10)
        assert row["frame_idx"] == 10
        assert row["timestamp_ms"] == 333
        assert row["is_keyframe"] == 1
        assert row["delta_score"] == 8.4
        db.close()


def test_insert_detections_and_query_by_frame():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        db.insert_frame(frame_idx=5, timestamp_ms=166, image_path="/tmp/5.jpg", is_keyframe=False, delta_score=0.1)
        db.insert_detection(
            frame_idx=5, track_id=1, class_name="person",
            x1=0.1, y1=0.2, x2=0.4, y2=0.9, confidence=0.92,
            mean_depth=None,
        )
        db.insert_detection(
            frame_idx=5, track_id=2, class_name="dog",
            x1=0.5, y1=0.6, x2=0.7, y2=0.85, confidence=0.81,
            mean_depth=None,
        )
        rows = db.get_detections_for_frame(5)
        assert len(rows) == 2
        classes = {r["class_name"] for r in rows}
        assert classes == {"person", "dog"}
        db.close()


def test_set_mean_depth_updates_detection():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        db.insert_frame(frame_idx=1, timestamp_ms=0, image_path="/tmp/1.jpg", is_keyframe=False, delta_score=0.0)
        det_id = db.insert_detection(
            frame_idx=1, track_id=1, class_name="person",
            x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=0.9,
            mean_depth=None,
        )
        db.set_detection_mean_depth(det_id, 0.42)
        rows = db.get_detections_for_frame(1)
        assert rows[0]["mean_depth"] == 0.42
        db.close()


def test_get_all_frame_indices_sorted():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        for idx in [10, 2, 7, 5]:
            db.insert_frame(frame_idx=idx, timestamp_ms=idx * 33, image_path=f"/tmp/{idx}.jpg", is_keyframe=False, delta_score=0.0)
        assert db.get_all_frame_indices() == [2, 5, 7, 10]
        db.close()
