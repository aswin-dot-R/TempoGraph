import tempfile
from pathlib import Path

from src.storage import TempoGraphDB
from src.modules.frame_scorer import FrameScorer


def test_scorer_prefers_detection_set_changes(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    # 3 frames; frame 1 has just a person, frame 2 has person+dog (new entity), frame 3 unchanged
    for idx in [0, 1, 2]:
        db.insert_frame(idx, idx * 1000, str(tmp_path / f"{idx}.jpg"), False, 0.5)

    db.insert_detection(0, 1, "person", 0.1, 0.1, 0.4, 0.9, 0.9)
    db.insert_detection(1, 1, "person", 0.1, 0.1, 0.4, 0.9, 0.9)
    db.insert_detection(1, 2, "dog", 0.5, 0.5, 0.7, 0.9, 0.85)
    db.insert_detection(2, 1, "person", 0.1, 0.1, 0.4, 0.9, 0.9)
    db.insert_detection(2, 2, "dog", 0.5, 0.5, 0.7, 0.9, 0.85)

    scorer = FrameScorer(db)
    # Force the scorer to pick top-2 (excluding mandatory keyframes for a clean test)
    top = scorer.score_and_select(
        candidate_frame_indices=[0, 1, 2],
        keyframe_indices=set(),
        k=2,
    )
    assert 1 in top  # frame 1 introduces the dog → must be picked
    db.close()


def test_scorer_always_includes_mandatory_keyframes(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    for idx in [0, 1, 2, 3, 4]:
        db.insert_frame(idx, idx * 1000, str(tmp_path / f"{idx}.jpg"), False, 0.0)
    # No detections at all -> scores will all be 0 except mandatory
    scorer = FrameScorer(db)
    top = scorer.score_and_select(
        candidate_frame_indices=[0, 1, 2, 3, 4],
        keyframe_indices={2, 4},
        k=3,
    )
    # Mandatory keyframes must appear regardless of K
    assert 2 in top
    assert 4 in top
    db.close()


def test_scorer_returns_sorted_indices(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    for idx in [10, 20, 30]:
        db.insert_frame(idx, idx, str(tmp_path / f"{idx}.jpg"), False, 0.0)
    scorer = FrameScorer(db)
    top = scorer.score_and_select(
        candidate_frame_indices=[10, 20, 30],
        keyframe_indices=set(),
        k=2,
    )
    assert top == sorted(top)
    db.close()
