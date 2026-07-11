"""Streamlit AppTest checks for ui/pages/Results.py on fixture run DBs.

Uses the TEMPOGRAPH_RESULTS_DIR override so the page renders a synthetic
run containing masked detections (and, for item 2, graph events).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rle import encode_to_string
from src.storage import TempoGraphDB

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PAGE = REPO_ROOT / "ui" / "pages" / "Results.py"


def make_fixture_run(results_dir: Path, name: str = "maskrun") -> Path:
    """Build a small run dir: frames on disk, masked detections, analysis."""
    import cv2

    run_dir = results_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    h, w = 90, 160
    db = TempoGraphDB(run_dir / "tempograph.db")
    rng = np.random.default_rng(0)
    for i, frame_idx in enumerate([0, 30, 60]):
        img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        p = frames_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(p), img)
        db.insert_frame(frame_idx, frame_idx * 33, str(p), i == 0, 0.5)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:40, 20 + i * 10:80 + i * 10] = 1
        db.insert_detection(
            frame_idx, 1, "dog", 0.12, 0.11, 0.55, 0.5, 0.91,
            mask_rle=encode_to_string(mask),
        )
        db.insert_detection(
            frame_idx, 2, "person", 0.5, 0.2, 0.9, 0.95, 0.83,
            mask_rle=encode_to_string(np.flipud(mask)),
        )
    # Pre-cache a summary so the page never probes the LLM endpoint.
    db.set_meta("summary", "Fixture run: a dog and a person interact.")
    db.close()

    analysis = {
        "entities": [
            {"id": "dog_1", "type": "dog", "first_seen": "00:00",
             "last_seen": "00:02", "description": "a brown dog"},
            {"id": "person_1", "type": "person", "first_seen": "00:00",
             "last_seen": "00:02", "description": "a person in a red coat"},
        ],
        "visual_events": [
            {"type": "approach", "start_time": "00:00", "end_time": "00:01",
             "description": "dog approaches person",
             "entities": ["dog_1", "person_1"], "confidence": 0.8},
            {"type": "interact", "start_time": "00:01", "end_time": "00:02",
             "description": "dog and person interact",
             "entities": ["dog_1", "person_1"], "confidence": 0.9},
        ],
        "audio_events": [],
        "summary": "A dog approaches and interacts with a person.",
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis))
    return run_dir


@pytest.fixture()
def fixture_results_dir(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    make_fixture_run(results_dir)
    monkeypatch.setenv("TEMPOGRAPH_RESULTS_DIR", str(results_dir))
    return results_dir


class TestFrameInspectorWithMasks:
    def test_page_renders_without_error_on_masked_db(self, fixture_results_dir):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(RESULTS_PAGE))
        at.run(timeout=60)
        assert not at.exception, f"page raised: {[e.value for e in at.exception]}"

    def test_masks_checkbox_present_and_enabled(self, fixture_results_dir):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(RESULTS_PAGE))
        at.run(timeout=60)
        cb = at.checkbox(key="insp_masks")
        assert cb is not None
        assert cb.value is True  # defaults on when the run has masks
        assert cb.disabled is False

    def test_inspector_rerenders_with_masks_toggled(self, fixture_results_dir):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(RESULTS_PAGE))
        at.run(timeout=60)
        at.checkbox(key="insp_masks").set_value(False)
        at.run(timeout=60)
        assert not at.exception
        at.checkbox(key="insp_masks").set_value(True)
        at.run(timeout=60)
        assert not at.exception
