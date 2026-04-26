from src.models import CameraMode, FrameSelectionResult, ChunkCaption


def test_camera_mode_enum_values():
    assert CameraMode.STATIC.value == "static"
    assert CameraMode.MOVING.value == "moving"
    assert CameraMode.AUTO.value == "auto"


def test_frame_selection_result_fields():
    r = FrameSelectionResult(
        frame_indices=[0, 5, 10],
        keyframe_indices=[0, 10],
        sampled_indices=[5],
        scan_indices=[0, 5, 10, 15],
        deltas=[0.0, 1.5, 12.3, 0.4],
        threshold=2.0,
        camera_mode=CameraMode.STATIC,
    )
    assert r.frame_indices == [0, 5, 10]
    assert r.keyframe_indices == [0, 10]
    assert r.threshold == 2.0


def test_chunk_caption_fields():
    c = ChunkCaption(
        chunk_id=0,
        frame_indices=[10, 20, 30],
        per_frame_lines={10: "person enters", 20: "(no change)", 30: "person sits"},
        summary="A person entered and sat down.",
        raw_response="FRAME_10: ...",
    )
    assert c.chunk_id == 0
    assert c.summary.startswith("A person")
