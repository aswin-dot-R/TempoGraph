"""Tests for dense captioning integration into the pipeline, aggregator, and UI.

Covers:
  1. The Dense captions stage in the pipeline (monkeypatched run_dense_captioning)
  2. The pipeline skips the stage when dense_captions=False
  3. CaptionAggregator.load_dense_timeline + aggregate() db_path integration
  4. The Results page renders with and without a frame_captions table
"""

from unittest.mock import MagicMock, patch
from pathlib import Path

import json
import numpy as np
import sqlite3
import pytest

import cv2

from src.models import PipelineConfig, AnalysisResult, CameraMode
from src.pipeline_v2 import PipelineV2
from src.storage import TempoGraphDB


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_video(path: Path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), 50 if i < 30 else 200, dtype=np.uint8)
        w.write(frame)
    w.release()


def _make_stage_callback() -> tuple:
    """Return a fresh (stages_list, on_stage_func) pair."""
    stages: list = []

    def on_stage(name: str, event: str, info: dict) -> None:
        stages.append((name, event, info))

    return stages, on_stage


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 – DenseCaptionStage
# ──────────────────────────────────────────────────────────────────────────────


class TestDenseCaptionStage:
    """Dense captioning stage in the pipeline: execution + resume guard."""

    def test_dense_captions_stage_runs_once(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_video(video)
        out = tmp_path / "out"

        stages, on_stage = _make_stage_callback()

        config = PipelineConfig(
            backend="llama-server",
            modules={
                "behavior": True,
                "detection": True,
                "depth": False,
                "audio": False,
            },
            fps=1.0,
            max_frames=20,
            confidence=0.5,
            video_path=str(video),
            output_dir=str(out),
        )

        with patch("src.pipeline_v2.ObjectDetector") as MockDet, patch(
            "src.pipeline_v2.LlamaServerBackend"
        ) as MockLLM, patch("src.pipeline_v2.CaptionAggregator") as MockAgg, patch(
            "src.pipeline_v2.run_dense_captioning"
        ) as MockDense:

            MockDet.return_value.detect_to_db = MagicMock()
            MockDet.return_value.cleanup = MagicMock()
            MockLLM.return_value.caption_chunks = MagicMock(return_value=[])
            MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
            MockAgg.return_value.aggregate = MagicMock(
                return_value=AnalysisResult(summary="ok")
            )

            def fake_run_dense(
                db_path,
                walker_url,
                verifier_url,
                on_progress,
                cancel_event,
                **kwargs,
            ):
                db = TempoGraphDB(db_path)
                try:
                    for idx in range(3):
                        db.insert_frame_caption(
                            frame_idx=idx,
                            caption=f"frame {idx}",
                            change_line=None,
                            walker_model="qwen",
                            escalated=False,
                        )
                finally:
                    db.close()
                return {
                    "walker": {
                        "captioned": 3,
                        "escalated": 0,
                        "skipped": 0,
                        "errors": 0,
                    },
                    "verifier": {
                        "verified": 0,
                        "agreed": 0,
                        "disagreed": 0,
                        "errors": 0,
                    },
                }

            MockDense.side_effect = fake_run_dense

            # First run: dense_captions=True → fake called, stage marked done
            pipeline = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                yolo_fps=1.0,
                vlm_fps=0.5,
                chunk_size=10,
                depth_enabled=False,
                use_segmentation=False,
                dense_captions=True,
                walker_url="http://127.0.0.1:8085",
                verifier_url="http://127.0.0.1:8096",
                on_stage=on_stage,
            )
            pipeline.run()

            assert MockDense.call_count == 1
            assert any(name == "Dense captions" for name, _, _ in stages)

            # Second run: resume guard → fake NOT called again,
            # stage is skipped (not re-executed).
            pipeline2 = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                yolo_fps=1.0,
                vlm_fps=0.5,
                chunk_size=10,
                depth_enabled=False,
                use_segmentation=False,
                dense_captions=True,
                walker_url="http://127.0.0.1:8085",
                verifier_url="http://127.0.0.1:8096",
                on_stage=on_stage,
            )
            pipeline2.run()
            assert MockDense.call_count == 1  # NOT called again
            # The "start" event fires before the resume check, so we expect
            # exactly one "done" and one "skipped" (not two "done"s).
            done_count = sum(
                1 for n, e, _ in stages if n == "Dense captions" and e == "done"
            )
            skipped_count = sum(
                1 for n, e, _ in stages if n == "Dense captions" and e == "skipped"
            )
            assert done_count == 1  # only first run does the work
            assert skipped_count == 1  # second run skips

    def test_dense_captions_skipped_when_flag_off(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_video(video)
        out = tmp_path / "out"

        stages, on_stage = _make_stage_callback()

        config = PipelineConfig(
            backend="llama-server",
            modules={
                "behavior": True,
                "detection": True,
                "depth": False,
                "audio": False,
            },
            fps=1.0,
            max_frames=20,
            confidence=0.5,
            video_path=str(video),
            output_dir=str(out),
        )

        with patch("src.pipeline_v2.ObjectDetector") as MockDet, patch(
            "src.pipeline_v2.LlamaServerBackend"
        ) as MockLLM, patch("src.pipeline_v2.CaptionAggregator") as MockAgg, patch(
            "src.pipeline_v2.run_dense_captioning"
        ) as MockDense:

            MockDet.return_value.detect_to_db = MagicMock()
            MockDet.return_value.cleanup = MagicMock()
            MockLLM.return_value.caption_chunks = MagicMock(return_value=[])
            MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
            MockAgg.return_value.aggregate = MagicMock(
                return_value=AnalysisResult(summary="ok")
            )
            # Dense is never called — we just need a mock object.
            MockDense.return_value = MagicMock()

            pipeline = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                yolo_fps=1.0,
                vlm_fps=0.5,
                chunk_size=10,
                depth_enabled=False,
                use_segmentation=False,
                dense_captions=False,
                walker_url="http://127.0.0.1:8085",
                verifier_url="http://127.0.0.1:8096",
                on_stage=on_stage,
            )
            pipeline.run()

        assert MockDense.call_count == 0
        # Only the "skipped" row is allowed — no start/done/progress/error.
        active_dense = [
            (n, e)
            for n, e, _ in stages
            if n == "Dense captions" and e not in ("skipped",)
        ]
        assert len(active_dense) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 – AggregatorDenseTimeline
# ──────────────────────────────────────────────────────────────────────────────


class TestAggregatorDenseTimeline:
    """CaptionAggregator.load_dense_timeline + aggregate() db_path."""

    def _build_fixture_db(self, tmp_path, db_path):
        """Insert 10 frame_caption rows into a fresh DB."""
        db = TempoGraphDB(db_path)
        # frames table (required for the JOIN in load_dense_timeline)
        for i in range(10):
            db.insert_frame(
                frame_idx=i,
                timestamp_ms=i * 1000,
                image_path=f"frame_{i}.jpg",
                is_keyframe=(i == 0),
                delta_score=0.5,
            )
        # frame_captions: frames 0,3,7 escalated
        for i in range(10):
            db.insert_frame_caption(
                frame_idx=i,
                caption=f"frame {i}",
                change_line=None,
                walker_model="qwen",
                escalated=(i in (0, 3, 7)),
            )
        # Frame 5: verifier disagrees → text = verifier_caption
        db.save_caption_verdict(
            frame_idx=5,
            verifier_caption="verified frame 5",
            verifier_agrees=False,
            verifier_model="35b",
        )
        # Frame 2: verifier agrees → text = walker caption (frame 2)
        db.save_caption_verdict(
            frame_idx=2,
            verifier_caption="verified frame 2",
            verifier_agrees=True,
            verifier_model="35b",
        )
        db.close()

    def test_load_dense_timeline(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        self._build_fixture_db(tmp_path, db_path)

        from src.aggregator import CaptionAggregator

        agg = CaptionAggregator()
        timeline = agg.load_dense_timeline(db_path, max_lines=5)

        # All 3 escalated rows always kept
        escalated = [t for t in timeline if t["escalated"]]
        assert len(escalated) == 3, f"expected 3 escalated, got {len(escalated)}"

        # Total ≤ max_lines
        assert len(timeline) <= 5

        # Sorted by timestamp
        timestamps = [t["timestamp_ms"] for t in timeline]
        assert timestamps == sorted(timestamps)

    def test_load_dense_timeline_keeps_verifier_disagreement_text(self, tmp_path):
        """With a generous max_lines, verifier disagreement text is preferred."""
        db_path = tmp_path / "db.sqlite"
        self._build_fixture_db(tmp_path, db_path)

        from src.aggregator import CaptionAggregator

        agg = CaptionAggregator()
        timeline = agg.load_dense_timeline(db_path, max_lines=100)

        # Frame 5: verifier_agrees=0 (disagreed) → use verifier_caption
        frame_5 = [t for t in timeline if t["timestamp_ms"] == 5000]
        assert len(frame_5) == 1
        assert frame_5[0]["text"] == "verified frame 5"

        # Frame 2: verifier_agrees=1 (agreed) → use walker caption
        frame_2 = [t for t in timeline if t["timestamp_ms"] == 2000]
        assert len(frame_2) == 1
        assert frame_2[0]["text"] == "frame 2"

        # Frame 1: no verifier → use walker caption
        frame_1 = [t for t in timeline if t["timestamp_ms"] == 1000]
        assert len(frame_1) == 1
        assert frame_1[0]["text"] == "frame 1"

    def test_aggregate_includes_dense_timeline_attribute(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        self._build_fixture_db(tmp_path, db_path)

        from src.aggregator import CaptionAggregator
        from src.models import ChunkCaption

        agg = CaptionAggregator()
        chunks = [
            ChunkCaption(
                chunk_id=0,
                frame_indices=[0, 1],
                per_frame_lines={"0": "line 0", "1": "line 1"},
                summary="chunk summary",
                raw_response="",
            )
        ]

        result = agg.aggregate(chunks, db_path=db_path)

        # The aggregator sets dense_timeline via setattr on a frozen model.
        # We assert it exists on the result instance (it was set by the aggregator).
        assert hasattr(result, "dense_timeline")
        assert result.dense_timeline is not None
        assert len(result.dense_timeline) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 – AppDenseCaptionRendering
# ──────────────────────────────────────────────────────────────────────────────


class TestAppDenseCaptionRendering:
    """Streamlit Results page with / without frame_captions."""

    def _insert_frame_captions(self, db: TempoGraphDB) -> None:
        db.insert_frame_caption(
            frame_idx=0,
            caption="frame 0",
            change_line=None,
            walker_model="qwen",
            escalated=False,
        )
        db.insert_frame_caption(
            frame_idx=1,
            caption="frame 1",
            change_line=None,
            walker_model="qwen",
            escalated=True,
        )
        db.save_caption_verdict(
            frame_idx=2,
            verifier_caption="disagreed frame 2",
            verifier_agrees=False,
            verifier_model="35b",
        )

    def _build_run_dir(
        self,
        tmp_path: Path,
        run_name: str,
        extra_insertions: list[callable] | None = None,
    ) -> Path:
        """Create a minimal run dir: frames + analysis.json + optional extras."""
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        frames_dir = run_dir / "frames"
        frames_dir.mkdir()

        db = TempoGraphDB(run_dir / "tempograph.db")
        rng = np.random.default_rng(0)
        for i, frame_idx in enumerate([0, 1, 2]):
            img = (rng.random((90, 160, 3)) * 255).astype(np.uint8)
            p = frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(p), img)
            db.insert_frame(
                frame_idx=frame_idx,
                timestamp_ms=frame_idx * 1000,
                image_path=str(p),
                is_keyframe=(frame_idx == 0),
                delta_score=0.5,
            )

        # Add audio_segments so _render_captions exercises the frame_captions path
        db2 = TempoGraphDB(run_dir / "tempograph.db")
        db2.insert_audio_segment(start_ms=0, end_ms=1000, text="hello world")
        db2.close()

        analysis = {
            "entities": [],
            "visual_events": [],
            "summary": f"Fixture run: {run_name}.",
        }
        (run_dir / "analysis.json").write_text(json.dumps(analysis))

        if extra_insertions is not None:
            for fn in extra_insertions:
                fn(db)
        db.close()

        return run_dir

    def test_page_renders_with_dense_captions(self, tmp_path, monkeypatch):
        from streamlit.testing.v1 import AppTest

        self._build_run_dir(
            tmp_path,
            "run_with_captions",
            extra_insertions=[self._insert_frame_captions],
        )
        monkeypatch.setenv("TEMPOGRAPH_RESULTS_DIR", str(tmp_path))

        at = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "ui" / "pages" / "Results.py")
        )
        at.run(timeout=60)
        assert not at.exception

    def test_page_renders_without_frame_captions(self, tmp_path, monkeypatch):
        from streamlit.testing.v1 import AppTest

        self._build_run_dir(tmp_path, "run_no_captions")
        monkeypatch.setenv("TEMPOGRAPH_RESULTS_DIR", str(tmp_path))

        at = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "ui" / "pages" / "Results.py")
        )
        at.run(timeout=60)
        assert not at.exception
