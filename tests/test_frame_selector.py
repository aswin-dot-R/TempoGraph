import numpy as np
import cv2
import tempfile
from pathlib import Path

from src.modules.frame_selector import FrameSelector
from src.models import CameraMode


def _make_synthetic_video(path: Path, n_frames: int = 60, w: int = 320, h: int = 240) -> None:
    """Create a synthetic video where frames 20-25 have a sudden brightness shift."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for i in range(n_frames):
        if 20 <= i < 25:
            frame = np.full((h, w, 3), 200, dtype=np.uint8)
        else:
            frame = np.full((h, w, 3), 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_static_mode_finds_keyframes_at_brightness_shift():
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "v.mp4"
        _make_synthetic_video(video)

        selector = FrameSelector()
        result = selector.select(
            video_path=str(video),
            camera_mode=CameraMode.STATIC,
            sample_fps=1.0,
            threshold_mult=1.0,
        )

        assert result.camera_mode == CameraMode.STATIC
        assert len(result.frame_indices) > 0
        # The frame at the brightness boundary should be a keyframe
        assert any(20 <= kf <= 25 for kf in result.keyframe_indices)


def test_static_mode_uniform_samples_at_user_fps():
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "v.mp4"
        _make_synthetic_video(video, n_frames=120)  # 4 seconds at 30fps

        selector = FrameSelector()
        result = selector.select(
            video_path=str(video),
            camera_mode=CameraMode.STATIC,
            sample_fps=1.0,
            threshold_mult=1.0,
        )

        # 1 Hz over 4 seconds -> roughly 4 sampled frames (give or take 1)
        assert 3 <= len(result.sampled_indices) <= 5


def test_returns_deltas_and_threshold_for_plotting():
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "v.mp4"
        _make_synthetic_video(video)

        result = FrameSelector().select(
            video_path=str(video), camera_mode=CameraMode.STATIC, sample_fps=1.0
        )

        assert len(result.deltas) == len(result.scan_indices)
        assert result.threshold > 0
