"""Estimate per-stage and total wall-clock cost for a v2 pipeline run.

Pure-Python heuristics calibrated for this machine (RTX 3060 + RX 9070 XT).
Used by the UI to show an ETA before the run actually starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2


# Empirical per-frame cost on the 3060 (seconds per frame), seg-variant adds ~15 %.
YOLO_S_PER_FRAME: Dict[str, float] = {
    "n": 0.025,
    "s": 0.045,
    "m": 0.080,
    "l": 0.130,
    "x": 0.220,
}

# Depth Anything V2 Small @ 3060 (~50 ms/frame); larger variants scale ~3-5x.
DEPTH_S_PER_FRAME = 0.10

# Whisper realtime ratios on a 3060 over Vulkan (transcription_seconds / audio_seconds).
WHISPER_RT_RATIO: Dict[str, float] = {
    "tiny":         1 / 32,
    "tiny.en":      1 / 32,
    "base":         1 / 16,
    "base.en":      1 / 16,
    "small":        1 / 6,
    "small.en":     1 / 6,
    "medium":       1 / 2,
    "medium.en":    1 / 2,
    "large-v1":     1.0,
    "large-v2":     1.0,
    "large-v3":     1.0,
    "large-v3-turbo": 1 / 4,
}

# VLM chunk cost on the 9070 XT (Qwen3.5-VL Q8_0 at chunk_size frames).
VLM_BASE_S_PER_CHUNK = 5.0
VLM_S_PER_FRAME_IN_CHUNK = 0.6

# Cold-start (model load) costs.
VLM_AUTOSTART_COLD_S = 12.0
WHISPER_LOAD_S = 0.5
YOLO_LOAD_S = 0.8
DEPTH_LOAD_S = 1.5

# Aggregator final synthesis call.
AGGREGATOR_S = 4.0

# Misc per-frame overhead (decode + jpeg write + DB inserts).
PER_FRAME_OVERHEAD_S = 0.015


@dataclass
class StageEstimate:
    name: str
    seconds: float
    note: str = ""


@dataclass
class RunEstimate:
    total_s: float
    stages: list  # list[StageEstimate]


def _video_meta(video_path: str) -> tuple:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    duration_s = n_frames_video / fps if fps > 0 else 0.0
    return fps, n_frames_video, duration_s


def estimate_run(
    video_path: str,
    yolo_fps: float,
    vlm_fps: float,
    chunk_size: int,
    yolo_size: str = "n",
    use_segmentation: bool = False,
    depth_enabled: bool = False,
    audio_enabled: bool = False,
    whisper_model: str = "base.en",
    vlm_frame_mode: str = "scored",
    keyframe_count_estimate: Optional[int] = None,
    vlm_autostart_cold: bool = True,
) -> RunEstimate:
    """Estimate total wall time. Heuristic — calibrate as you collect runs."""
    _fps, _n_video_frames, duration_s = _video_meta(video_path)

    n_sampled = max(1, int(round(duration_s * yolo_fps)))

    if vlm_frame_mode == "keyframes" and keyframe_count_estimate is not None:
        n_vlm = max(1, keyframe_count_estimate)
    else:
        n_vlm = max(1, int(round(duration_s * vlm_fps)))

    n_chunks = (n_vlm + chunk_size - 1) // chunk_size

    stages: list = []

    # 1. Frame selection (motion delta scan over the entire video at sample fps)
    frame_select_s = n_sampled * 0.02 + n_sampled * PER_FRAME_OVERHEAD_S
    stages.append(StageEstimate("Frame selection", frame_select_s,
                                f"{n_sampled} sampled frames"))

    # 1.5 Audio
    if audio_enabled:
        ratio = WHISPER_RT_RATIO.get(whisper_model, 1 / 8)
        audio_s = WHISPER_LOAD_S + duration_s * ratio
        stages.append(StageEstimate("Audio transcription", audio_s,
                                    f"{whisper_model} on {duration_s:.1f}s of audio"))

    # 2. YOLO sweep
    per = YOLO_S_PER_FRAME.get(yolo_size, 0.05)
    if use_segmentation:
        per *= 1.15
    yolo_s = YOLO_LOAD_S + n_sampled * per
    stages.append(StageEstimate("YOLO detection", yolo_s,
                                f"yolo26{yolo_size}{'-seg' if use_segmentation else ''} × {n_sampled}"))

    # 3. Depth
    if depth_enabled:
        depth_s = DEPTH_LOAD_S + n_sampled * DEPTH_S_PER_FRAME
        stages.append(StageEstimate("Depth estimation", depth_s,
                                    f"vits × {n_sampled}"))

    # 4. Frame scoring (cheap)
    stages.append(StageEstimate("Frame scoring", 0.5,
                                f"{n_vlm} VLM frames"))

    # 5. VLM captioning
    if vlm_autostart_cold:
        stages.append(StageEstimate("VLM autostart", VLM_AUTOSTART_COLD_S,
                                    "qwen3.5-9B model load on AMD"))
    chunk_s = VLM_BASE_S_PER_CHUNK + chunk_size * VLM_S_PER_FRAME_IN_CHUNK
    vlm_s = n_chunks * chunk_s
    stages.append(StageEstimate("VLM captioning", vlm_s,
                                f"{n_chunks} chunks × ~{chunk_s:.1f}s"))

    # 6. Aggregator
    stages.append(StageEstimate("Aggregation", AGGREGATOR_S,
                                "single text-only LLM call"))

    total = sum(s.seconds for s in stages)
    return RunEstimate(total_s=total, stages=stages)


def format_seconds(s: float) -> str:
    s = max(0, int(s))
    m, sec = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"
