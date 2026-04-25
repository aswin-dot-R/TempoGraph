import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.storage import TempoGraphDB
from src.modules.detector import ObjectDetector


def _make_jpg(path: Path) -> None:
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_detect_to_db_inserts_rows(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        f1 = td / "f1.jpg"
        f2 = td / "f2.jpg"
        _make_jpg(f1)
        _make_jpg(f2)

        db_path = td / "t.db"
        db = TempoGraphDB(db_path)
        db.insert_frame(0, 0, str(f1), True, 0.0)
        db.insert_frame(1, 1000, str(f2), False, 0.0)

        # Mock YOLO model: return one box per frame
        fake_box = MagicMock()
        fake_box.xyxy = [np.array([10.0, 20.0, 100.0, 200.0])]
        fake_box.conf = [0.85]
        fake_box.cls = [0]

        fake_boxes = MagicMock()
        fake_boxes.__iter__ = lambda self: iter([fake_box])
        fake_boxes.__len__ = MagicMock(return_value=1)
        fake_boxes.id = None

        fake_result = MagicMock()
        fake_result.boxes = fake_boxes

        fake_model = MagicMock()
        fake_model.names = {0: "person"}
        fake_model.track = MagicMock(return_value=[fake_result])

        det = ObjectDetector(device="cpu")
        det._model = fake_model

        det.detect_to_db(
            frame_indices=[0, 1],
            frame_paths=[f1, f2],
            db=db,
            frame_width=320,
            frame_height=240,
        )

        rows0 = db.get_detections_for_frame(0)
        rows1 = db.get_detections_for_frame(1)
        assert len(rows0) == 1
        assert len(rows1) == 1
        # Coords should be normalized into [0,1]
        assert 0.0 <= rows0[0]["x1"] <= 1.0
        assert 0.0 <= rows0[0]["y2"] <= 1.0
        assert rows0[0]["class_name"] == "person"
        db.close()
