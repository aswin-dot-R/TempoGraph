"""Integration tests for resume-from-DB stage guards.

Verifies:
- A run DB with a stage marked complete skips that stage when the pipeline runs.
- A fresh DB runs all stages.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import PipelineConfig, CameraMode, AnalysisResult
from src.pipeline_v2 import PipelineV2
from src.storage import TempoGraphDB


# ── fixtures ─────────────────────────────────────────────────────────────


def _make_video(path: Path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), 50 if i < 30 else 200, dtype=np.uint8)
        w.write(frame)
    w.release()


def _make_run_db_with_stage(out_dir: Path, stage_name: str) -> Path:
    """Create a DB with *stage_name* pre-marked complete, plus a frame row."""
    db = TempoGraphDB(out_dir / "tempograph.db")
    db.insert_frame(
        frame_idx=5,
        timestamp_ms=166,
        image_path=str(out_dir / "frames" / "frame_000005.jpg"),
        is_keyframe=False,
        delta_score=0.1,
    )
    db.mark_stage_complete(stage_name, elapsed_s=1.0, n_units=1)
    db.close()
    return out_dir / "tempograph.db"


# ── tests ────────────────────────────────────────────────────────────────


class TestResumeSkipsCompletedStage:
    """A run DB where Frame selection is marked complete should skip it."""

    def test_frame_selection_skipped(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_video(video)
        out = tmp_path / "out"
        out.mkdir()
        frames_dir = out / "frames"
        frames_dir.mkdir()

        # Pre-mark Frame selection as complete
        _make_run_db_with_stage(out, "Frame selection")

        # Ensure the saved frame path exists so resume can load it
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000005.jpg"), fake_frame)

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

        with patch("src.pipeline_v2.FrameSelector") as MockFS, patch(
            "src.pipeline_v2.ObjectDetector"
        ) as MockDet, patch("src.pipeline_v2.LlamaServerBackend") as MockLLM, patch(
            "src.pipeline_v2.CaptionAggregator"
        ) as MockAgg, patch(
            "src.pipeline_v2.FrameScorer"
        ) as MockScorer:
            MockFS.return_value.select.return_value = MagicMock(
                frame_indices=[5],
                keyframe_indices=[],
                scan_indices=[5],
                deltas=[0.1],
            )
            MockDet.return_value.detect_to_db = MagicMock()
            MockDet.return_value.cleanup = MagicMock()
            MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
            MockAgg.return_value.aggregate = MagicMock(
                return_value=AnalysisResult(summary="ok")
            )
            MockScorer.return_value.score_and_select = MagicMock(return_value=[5])

            pipeline = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                depth_enabled=False,
                skip_vlm=False,
            )
            result = pipeline.run()

        # Frame selection's mock should NOT be called (stage was skipped)
        MockFS.assert_not_called()
        # But later stages SHOULD be called
        MockDet.return_value.detect_to_db.assert_called_once()
        MockLLM.return_value.caption_frames_dynamic.assert_called_once()
        MockAgg.return_value.aggregate.assert_called_once()

        # All stages should be marked complete after a fresh run
        db = TempoGraphDB(out / "tempograph.db")
        assert db.is_stage_complete("YOLO detection")
        assert db.is_stage_complete("Frame scoring")
        assert db.is_stage_complete("VLM captioning")
        assert db.is_stage_complete("Aggregation")
        db.close()

        assert result.analysis.summary == "ok"


class TestFreshRunRunsAllStages:
    """A fresh DB should run every stage."""

    def test_all_stages_run_fresh(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_video(video)
        out = tmp_path / "out"
        out.mkdir()
        frames_dir = out / "frames"
        frames_dir.mkdir()

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

        with patch("src.pipeline_v2.FrameSelector") as MockFS, patch(
            "src.pipeline_v2.ObjectDetector"
        ) as MockDet, patch("src.pipeline_v2.LlamaServerBackend") as MockLLM, patch(
            "src.pipeline_v2.CaptionAggregator"
        ) as MockAgg, patch(
            "src.pipeline_v2.FrameScorer"
        ) as MockScorer:
            MockFS.return_value.select.return_value = MagicMock(
                frame_indices=[5],
                keyframe_indices=[],
                scan_indices=[5],
                deltas=[0.1],
            )
            MockDet.return_value.detect_to_db = MagicMock()
            MockDet.return_value.cleanup = MagicMock()
            MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
            MockAgg.return_value.aggregate = MagicMock(
                return_value=AnalysisResult(summary="ok")
            )
            MockScorer.return_value.score_and_select = MagicMock(return_value=[5])

            pipeline = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                depth_enabled=False,
                skip_vlm=False,
            )
            result = pipeline.run()

        # Every stage mock should have been called
        MockFS.assert_called_once()
        MockDet.return_value.detect_to_db.assert_called_once()
        MockLLM.return_value.caption_frames_dynamic.assert_called_once()
        MockAgg.return_value.aggregate.assert_called_once()

        # All stages should be marked complete
        db = TempoGraphDB(out / "tempograph.db")
        for stage in [
            "Frame selection",
            "YOLO detection",
            "Frame scoring",
            "VLM captioning",
            "Aggregation",
        ]:
            assert db.is_stage_complete(stage), f"{stage} not marked complete"
        db.close()

        assert result.analysis.summary == "ok"


class TestResumeSkipsIntermediateStage:
    """If YOLO detection is complete, it should be skipped."""

    def test_yolo_skipped(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_video(video)
        out = tmp_path / "out"
        out.mkdir()
        frames_dir = out / "frames"
        frames_dir.mkdir()

        # Pre-mark Frame selection and YOLO detection as complete
        db = TempoGraphDB(out / "tempograph.db")
        db.insert_frame(
            frame_idx=5,
            timestamp_ms=166,
            image_path=str(frames_dir / "frame_000005.jpg"),
            is_keyframe=False,
            delta_score=0.1,
        )
        db.mark_stage_complete("Frame selection", elapsed_s=1.0, n_units=1)
        db.mark_stage_complete("YOLO detection", elapsed_s=2.0, n_units=10)
        db.close()

        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000005.jpg"), fake_frame)

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

        with patch("src.pipeline_v2.FrameSelector") as MockFS, patch(
            "src.pipeline_v2.ObjectDetector"
        ) as MockDet, patch("src.pipeline_v2.LlamaServerBackend") as MockLLM, patch(
            "src.pipeline_v2.CaptionAggregator"
        ) as MockAgg, patch(
            "src.pipeline_v2.FrameScorer"
        ) as MockScorer:
            MockFS.return_value.select.return_value = MagicMock(
                frame_indices=[5],
                keyframe_indices=[],
                scan_indices=[5],
                deltas=[0.1],
            )
            MockDet.return_value.detect_to_db = MagicMock()
            MockDet.return_value.cleanup = MagicMock()
            MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
            MockAgg.return_value.aggregate = MagicMock(
                return_value=AnalysisResult(summary="ok")
            )
            MockScorer.return_value.score_and_select = MagicMock(return_value=[5])

            pipeline = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                depth_enabled=False,
                skip_vlm=False,
            )
            result = pipeline.run()

        # YOLO detection's mock should NOT be called (stage was skipped)
        MockDet.assert_not_called()
        # Frame selection also skipped
        MockFS.assert_not_called()
        # But downstream stages SHOULD be called
        MockLLM.return_value.caption_frames_dynamic.assert_called_once()
        MockAgg.return_value.aggregate.assert_called_once()

        # Check completed stages
        db = TempoGraphDB(out / "tempograph.db")
        assert db.is_stage_complete("Frame selection")
        assert db.is_stage_complete("YOLO detection")
        assert db.is_stage_complete("Frame scoring")
        assert db.is_stage_complete("VLM captioning")
        assert db.is_stage_complete("Aggregation")
        db.close()

        assert result.analysis.summary == "ok"


class TestResumeLoadsAnalysisJsonAsPydantic:
    """When Aggregation is already complete and analysis.json exists, the pipeline
    must load it into a real AnalysisResult instance, not a duck-typed object.
    """

    def test_resume_returns_real_analysisresult(self, tmp_path):
        from src.models import Entity, VisualEvent, AudioEvent, BehaviorType, SoundType

        video = tmp_path / "v.mp4"
        _make_video(video)
        out = tmp_path / "out"
        out.mkdir()
        frames_dir = out / "frames"
        frames_dir.mkdir()

        # Pre-mark all stages as complete
        db = TempoGraphDB(out / "tempograph.db")
        db.insert_frame(
            frame_idx=5,
            timestamp_ms=166,
            image_path=str(frames_dir / "frame_000005.jpg"),
            is_keyframe=False,
            delta_score=0.1,
        )
        for stage in [
            "Frame selection",
            "Audio transcription",
            "YOLO detection",
            "Depth estimation",
            "Frame scoring",
            "Dense captions",
            "VLM captioning",
            "Aggregation",
        ]:
            db.mark_stage_complete(stage, elapsed_s=1.0, n_units=1)
        db.close()

        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000005.jpg"), fake_frame)

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

        # Write a valid analysis.json with proper Pydantic-serializable data
        import json

        sample_data = {
            "entities": [
                {
                    "id": "e1",
                    "type": "person",
                    "description": "A person in red shirt",
                    "first_seen": "00:05",
                    "last_seen": "00:15",
                }
            ],
            "visual_events": [
                {
                    "type": "approach",
                    "entities": ["e1"],
                    "start_time": "00:05",
                    "end_time": "00:10",
                    "description": "Person approaches table",
                    "confidence": 0.85,
                }
            ],
            "audio_events": [
                {
                    "type": "speech",
                    "start_time": "00:06",
                    "end_time": "00:09",
                    "speaker": " narrator",
                    "text": "The person sits down.",
                    "confidence": 0.9,
                }
            ],
            "summary": "Test summary",
        }

        (out / "analysis.json").write_text(json.dumps(sample_data))

        # Mock out stages that would be skipped anyway
        with patch("src.pipeline_v2.FrameSelector") as MockFS, patch(
            "src.pipeline_v2.ObjectDetector"
        ) as MockDet, patch("src.pipeline_v2.LlamaServerBackend") as MockLLM, patch(
            "src.pipeline_v2.CaptionAggregator"
        ) as MockAgg, patch(
            "src.pipeline_v2.FrameScorer"
        ) as MockScorer, patch(
            "src.pipeline_v2.WhisperCppTranscriber"
        ) as MockWhisper:
            MockFS.return_value.select.return_value = MagicMock(
                frame_indices=[5],
                keyframe_indices=[],
                scan_indices=[5],
                deltas=[0.1],
            )
            MockDet.return_value.detect_to_db = MagicMock()
            MockDet.return_value.cleanup = MagicMock()
            MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
            MockAgg.return_value.aggregate = MagicMock(
                return_value=AnalysisResult(summary="ok")
            )
            MockScorer.return_value.score_and_select = MagicMock(return_value=[5])
            MockWhisper.return_value.transcribe = MagicMock(return_value=[])

            pipeline = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                depth_enabled=False,
                skip_vlm=False,
            )
            result = pipeline.run()

        # The returned analysis must be a real AnalysisResult instance
        assert isinstance(result.analysis, AnalysisResult)
        assert result.analysis.summary == "Test summary"

        # Entities must be real Entity instances, not dicts
        assert len(result.analysis.entities) == 1
        entity = result.analysis.entities[0]
        assert isinstance(entity, Entity)
        assert entity.id == "e1"
        assert entity.type == "person"

        # Visual events must be real VisualEvent instances
        assert len(result.analysis.visual_events) == 1
        ve = result.analysis.visual_events[0]
        assert isinstance(ve, VisualEvent)
        assert ve.type == BehaviorType.APPROACH
        assert ve.description == "Person approaches table"

        # Audio events must be real AudioEvent instances
        assert len(result.analysis.audio_events) == 1
        ae = result.analysis.audio_events[0]
        assert isinstance(ae, AudioEvent)
        assert ae.type == SoundType.SPEECH
        assert ae.text == "The person sits down."

    def test_corrupt_analysis_json_does_not_crash(self, tmp_path):
        """A corrupt analysis.json should not crash the pipeline."""
        video = tmp_path / "v.mp4"
        _make_video(video)
        out = tmp_path / "out"
        out.mkdir()
        frames_dir = out / "frames"
        frames_dir.mkdir()

        # Pre-mark all stages as complete
        db = TempoGraphDB(out / "tempograph.db")
        db.insert_frame(
            frame_idx=5,
            timestamp_ms=166,
            image_path=str(frames_dir / "frame_000005.jpg"),
            is_keyframe=False,
            delta_score=0.1,
        )
        for stage in [
            "Frame selection",
            "Audio transcription",
            "YOLO detection",
            "Depth estimation",
            "Frame scoring",
            "Dense captions",
            "VLM captioning",
            "Aggregation",
        ]:
            db.mark_stage_complete(stage, elapsed_s=1.0, n_units=1)
        db.close()

        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000005.jpg"), fake_frame)

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

        # Write a corrupt analysis.json (invalid JSON)
        (out / "analysis.json").write_text("not valid json {{{")

        # Mock out stages that would be skipped anyway
        with patch("src.pipeline_v2.FrameSelector") as MockFS, patch(
            "src.pipeline_v2.ObjectDetector"
        ) as MockDet, patch("src.pipeline_v2.LlamaServerBackend") as MockLLM, patch(
            "src.pipeline_v2.CaptionAggregator"
        ) as MockAgg, patch(
            "src.pipeline_v2.FrameScorer"
        ) as MockScorer, patch(
            "src.pipeline_v2.WhisperCppTranscriber"
        ) as MockWhisper:
            MockFS.return_value.select.return_value = MagicMock(
                frame_indices=[5],
                keyframe_indices=[],
                scan_indices=[5],
                deltas=[0.1],
            )
            MockDet.return_value.detect_to_db = MagicMock()
            MockDet.return_value.cleanup = MagicMock()
            MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
            MockAgg.return_value.aggregate = MagicMock(
                return_value=AnalysisResult(summary="ok")
            )
            MockScorer.return_value.score_and_select = MagicMock(return_value=[5])
            MockWhisper.return_value.transcribe = MagicMock(return_value=[])

            pipeline = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                depth_enabled=False,
                skip_vlm=False,
            )
            result = pipeline.run()

        # The pipeline should not crash and should return None for analysis
        assert result.analysis is None

    def test_nonconformant_analysis_json_does_not_crash(self, tmp_path):
        """A non-conformant analysis.json (valid JSON but wrong schema) should not crash."""
        video = tmp_path / "v.mp4"
        _make_video(video)
        out = tmp_path / "out"
        out.mkdir()
        frames_dir = out / "frames"
        frames_dir.mkdir()

        # Pre-mark all stages as complete
        db = TempoGraphDB(out / "tempograph.db")
        db.insert_frame(
            frame_idx=5,
            timestamp_ms=166,
            image_path=str(frames_dir / "frame_000005.jpg"),
            is_keyframe=False,
            delta_score=0.1,
        )
        for stage in [
            "Frame selection",
            "Audio transcription",
            "YOLO detection",
            "Depth estimation",
            "Frame scoring",
            "Dense captions",
            "VLM captioning",
            "Aggregation",
        ]:
            db.mark_stage_complete(stage, elapsed_s=1.0, n_units=1)
        db.close()

        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000005.jpg"), fake_frame)

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

        # Write a non-conformant analysis.json (valid JSON but wrong schema)
        import json

        bad_data = {"entities": "not a list", "summary": 123}  # wrong types
        (out / "analysis.json").write_text(json.dumps(bad_data))

        # Mock out stages that would be skipped anyway
        with patch("src.pipeline_v2.FrameSelector") as MockFS, patch(
            "src.pipeline_v2.ObjectDetector"
        ) as MockDet, patch("src.pipeline_v2.LlamaServerBackend") as MockLLM, patch(
            "src.pipeline_v2.CaptionAggregator"
        ) as MockAgg, patch(
            "src.pipeline_v2.FrameScorer"
        ) as MockScorer, patch(
            "src.pipeline_v2.WhisperCppTranscriber"
        ) as MockWhisper:
            MockFS.return_value.select.return_value = MagicMock(
                frame_indices=[5],
                keyframe_indices=[],
                scan_indices=[5],
                deltas=[0.1],
            )
            MockDet.return_value.detect_to_db = MagicMock()
            MockDet.return_value.cleanup = MagicMock()
            MockLLM.return_value.caption_frames_dynamic = MagicMock(return_value=[])
            MockAgg.return_value.aggregate = MagicMock(
                return_value=AnalysisResult(summary="ok")
            )
            MockScorer.return_value.score_and_select = MagicMock(return_value=[5])
            MockWhisper.return_value.transcribe = MagicMock(return_value=[])

            pipeline = PipelineV2(
                config,
                camera_mode=CameraMode.STATIC,
                depth_enabled=False,
                skip_vlm=False,
            )
            result = pipeline.run()

        # The pipeline should not crash and should return None for analysis
        assert result.analysis is None
