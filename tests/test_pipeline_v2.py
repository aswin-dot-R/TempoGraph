from unittest.mock import patch, MagicMock
from pathlib import Path

import cv2
import numpy as np

from src.models import PipelineConfig, AnalysisResult, CameraMode
from src.pipeline_v2 import PipelineV2


def _make_video(path: Path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), 50 if i < 30 else 200, dtype=np.uint8)
        w.write(frame)
    w.release()


def test_pipeline_v2_runs_end_to_end_with_mocked_models(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video)
    out = tmp_path / "out"

    config = PipelineConfig(
        backend="llama-server",
        modules={"behavior": True, "detection": True, "depth": False, "audio": False},
        fps=1.0,
        max_frames=20,
        confidence=0.5,
        video_path=str(video),
        output_dir=str(out),
    )

    with patch("src.pipeline_v2.ObjectDetector") as MockDet, \
         patch("src.pipeline_v2.LlamaServerBackend") as MockLLM, \
         patch("src.pipeline_v2.CaptionAggregator") as MockAgg:
        MockDet.return_value.detect_to_db = MagicMock()
        MockDet.return_value.cleanup = MagicMock()
        MockLLM.return_value.caption_chunks = MagicMock(return_value=[])
        MockAgg.return_value.aggregate = MagicMock(
            return_value=AnalysisResult(summary="ok")
        )

        pipeline = PipelineV2(
            config,
            camera_mode=CameraMode.STATIC,
            yolo_fps=1.0,
            vlm_fps=0.5,
            chunk_size=10,
            depth_enabled=False,
            use_segmentation=False,
        )
        result = pipeline.run()

    assert result.analysis.summary == "ok"
    assert (out / "tempograph.db").exists()
    MockDet.return_value.detect_to_db.assert_called_once()
    MockLLM.return_value.caption_chunks.assert_called_once()
    MockAgg.return_value.aggregate.assert_called_once()
