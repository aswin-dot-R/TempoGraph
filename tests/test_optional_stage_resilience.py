"""Tests for graceful degradation of optional pipeline stages.

Contract: when an optional stage (depth, dense captions, audio) fails,
the pipeline emits an error event, writes no run_stages row for that
stage, and returns a result. The derive_plan function must report
honesty about ``importlib.util.find_spec("transformers")`` availability.
"""

from unittest.mock import MagicMock, patch

import pytest
import sqlite3

import cv2
import numpy as np

from src.models import PipelineConfig, AnalysisResult, CameraMode
from src.pipeline_v2 import PipelineV2
from src.auto_profile import derive_plan, VideoFacts


# ── fixtures ───────────────────────────────────────────────────────


def _make_video(path):
    """Build a tiny real mp4 (60 frames, 320x240, 30 fps)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), 50 if i < 30 else 200, dtype=np.uint8)
        w.write(frame)
    w.release()


# ── helpers ────────────────────────────────────────────────────────


def _run_resilience(
    video,
    out,
    *,
    depth_enabled=False,
    dense_captions=False,
    audio_enabled=False,
    whisper_binary=None,
    depth_patch=None,
    dense_patch=None,
    whisper_patch=None,
):
    """Run the pipeline with the given optional stages enabled.

    Applies patches for the three optional-stage entry points so each
    test can inject success or failure independently.  Returns
    ``(result, events)``.  ``result`` is ``None`` if the pipeline raised
    an unhandled exception.
    """
    config = PipelineConfig(
        backend="llama-server",
        modules={
            "behavior": True,
            "detection": True,
            "depth": depth_enabled,
            "audio": audio_enabled,
        },
        fps=1.0,
        max_frames=20,
        confidence=0.5,
        video_path=str(video),
        output_dir=str(out),
        whisper_binary=whisper_binary,
    )

    events = []

    with patch("src.pipeline_v2.ObjectDetector") as MockDet, patch(
        "src.pipeline_v2.LlamaServerBackend"
    ) as MockLLM, patch("src.pipeline_v2.CaptionAggregator") as MockAgg, patch(
        "src.pipeline_v2.run_dense_captioning", side_effect=dense_patch
    ), patch(
        "src.pipeline_v2.WhisperCppTranscriber", side_effect=whisper_patch
    ), patch(
        "src.pipeline_v2.DepthEstimator", side_effect=depth_patch
    ):

        MockDet.return_value.detect_to_db = MagicMock()
        MockDet.return_value.cleanup = MagicMock()
        MockLLM.return_value.caption_chunks = MagicMock(return_value=[])
        MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
        MockAgg.return_value.aggregate = MagicMock(
            return_value=AnalysisResult(summary="ok")
        )

        pipeline = PipelineV2(
            config,
            camera_mode=CameraMode.STATIC,
            yolo_fps=1.0,
            vlm_fps=0.5,
            chunk_size=10,
            depth_enabled=depth_enabled,
            dense_captions=dense_captions,
            use_segmentation=False,
            audio_enabled=audio_enabled,
            whisper_binary=whisper_binary,
            on_stage=lambda n, e, i: events.append((n, e, i)),
        )

        try:
            result = pipeline.run()
        except Exception:
            result = None

        return result, events


def _stages(out):
    """Return list of stage_name strings from the ``run_stages`` table."""
    db_path = out / "tempograph.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        return [
            row[0]
            for row in conn.execute("SELECT stage_name FROM run_stages").fetchall()
        ]
    finally:
        conn.close()


# ── Optional-stage resilience ──────────────────────────────────────


def test_depth_importerror_doesnt_kill_the_run(tmp_path):
    """DepthEstimator constructor raises ImportError — pipeline survives."""
    video = tmp_path / "v.mp4"
    _make_video(video)
    out = tmp_path / "out"
    out.mkdir()

    result, events = _run_resilience(
        video,
        out,
        depth_enabled=True,
        depth_patch=ImportError("transformers is required for depth estimation"),
    )

    assert result is not None
    assert any(e[0] == "Depth estimation" and e[1] == "error" for e in events)
    assert "Depth estimation" not in _stages(out)


def test_dense_captioning_failure_doesnt_kill_the_run(tmp_path):
    """run_dense_captioning raises — pipeline survives."""
    video = tmp_path / "v.mp4"
    _make_video(video)
    out = tmp_path / "out"
    out.mkdir()

    result, events = _run_resilience(
        video,
        out,
        dense_captions=True,
        dense_patch=RuntimeError("verifier down"),
    )

    assert result is not None
    assert any(e[0] == "Dense captions" and e[1] == "error" for e in events)
    assert "Dense captions" not in _stages(out)


def test_whisper_constructor_failure_doesnt_kill_the_run(tmp_path):
    """WhisperCppTranscriber constructor raises — pipeline survives."""
    video = tmp_path / "v.mp4"
    _make_video(video)
    out = tmp_path / "out"
    out.mkdir()

    result, events = _run_resilience(
        video,
        out,
        audio_enabled=True,
        whisper_binary="/nonexistent/whisper-cli",
        whisper_patch=FileNotFoundError("no whisper-cli"),
    )

    assert result is not None
    assert any(e[0] == "Audio transcription" and e[1] == "error" for e in events)
    assert "Audio transcription" not in _stages(out)


# ── derive_plan honesty ───────────────────────────────────────────


def test_derive_plan_honesty_unavailable():
    """transformers not installed → depth_enabled=False, note added."""
    with patch("src.auto_profile.importlib.util.find_spec", return_value=None):
        plan = derive_plan(
            VideoFacts(
                duration_s=30.5, width=1920, height=1080, fps=30.0, has_audio=True
            )
        )

    assert plan.depth_enabled is False
    assert "depth off: transformers not installed" in plan.notes


def test_derive_plan_honesty_available():
    """transformers installed → depth_enabled follows size rule (True for 1080p)."""
    with patch("src.auto_profile.importlib.util.find_spec", return_value=MagicMock()):
        plan = derive_plan(
            VideoFacts(
                duration_s=30.5, width=1920, height=1080, fps=30.0, has_audio=True
            )
        )

    assert plan.depth_enabled is True
    assert "depth off: transformers not installed" not in plan.notes


# ── Happy path ────────────────────────────────────────────────────


def test_happy_path_unchanged(tmp_path):
    """All optional stages succeed → no error events at all."""
    video = tmp_path / "v.mp4"
    _make_video(video)
    out = tmp_path / "out"
    out.mkdir()

    result, events = _run_resilience(
        video,
        out,
        depth_enabled=True,
        dense_captions=True,
        audio_enabled=True,
        depth_patch=lambda **kw: MagicMock(
            estimate_to_db=MagicMock(),
            cleanup=MagicMock(),
        ),
        dense_patch=lambda **kw: {
            "walker": {"captioned": 0, "escalated": 0},
            "verifier": {},
        },
        whisper_patch=lambda **kw: MagicMock(transcribe_video=lambda path: []),
    )

    assert result is not None
    assert all(e[1] != "error" for e in events)
