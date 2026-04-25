import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from src.storage import TempoGraphDB
from src.modules.depth import DepthEstimator


def _make_jpg(path: Path) -> None:
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_estimate_to_db_writes_depth_and_per_bbox_means(tmp_path):
    f1 = tmp_path / "f1.jpg"
    _make_jpg(f1)

    db = TempoGraphDB(tmp_path / "t.db")
    db.insert_frame(0, 0, str(f1), True, 0.0)
    det_id = db.insert_detection(
        frame_idx=0, track_id=1, class_name="person",
        x1=0.0, y1=0.0, x2=0.5, y2=0.5, confidence=0.9,
    )

    # Mock depth model: returns a depth map where left-half=0.2, right-half=0.8
    fake_depth = np.zeros((240, 320), dtype=np.float32)
    fake_depth[:, :160] = 0.2
    fake_depth[:, 160:] = 0.8

    fake_model = MagicMock()
    fake_model.infer_image = MagicMock(return_value=fake_depth)

    estimator = DepthEstimator(device="cpu")
    estimator._model = fake_model

    estimator.estimate_to_db(
        frame_indices=[0],
        frame_paths=[f1],
        db=db,
        output_dir=str(tmp_path / "depth"),
    )

    # Depth file written
    assert db.get_depth_path(0) is not None
    # Per-bbox mean depth populated; bbox covers top-left quadrant (left half) -> ~0.2
    rows = db.get_detections_for_frame(0)
    assert rows[0]["mean_depth"] is not None
    assert abs(rows[0]["mean_depth"] - 0.2) < 0.05
    db.close()
