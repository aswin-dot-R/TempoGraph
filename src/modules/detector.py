"""
YOLO11 detection module.

Model: yolo11n.pt (~6MB) or yolo11s.pt (~22MB) - latest from Ultralytics
VRAM: ~0.5GB for yolo11n, ~1GB for yolo11s
Processing time: ~50-100ms per frame on GPU

Available models:
- yolo11n.pt - nano (fastest, lowest accuracy)
- yolo11s.pt - small (balanced)
- yolo11m.pt - medium (better accuracy)
- yolo11l.pt - large (best accuracy, slower)
- yolo11x.pt - extra large (highest accuracy, slowest)

Usage:
from ultralytics import YOLO
model = YOLO("yolo11s.pt")

# Detection on single frame:
results = model(frame, conf=0.5, imgsz=640, verbose=False)
# results[0].boxes.xyxy  → tensor of [x1, y1, x2, x2]
# results[0].boxes.conf  → tensor of confidences
# results[0].boxes.cls   → tensor of class indices
# model.names[int(cls)]  → class name string

# For tracking across frames (assigns persistent IDs):
results = model.track(frame, conf=0.5, imgsz=640, persist=True, verbose=False)
# results[0].boxes.id  → tensor of track IDs (or None if tracking fails)

VRAM management:
- Load model to CUDA
- Process all frames
- del model + torch.cuda.empty_cache()
- Must free VRAM before Qwen loads
"""

import time
import torch
import gc
import logging
from pathlib import Path
from typing import List, Optional
from src.models import DetectionResult, DetectionBox


class ObjectDetector:
    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.5,
        imgsz: int = 640,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.logger = logging.getLogger(__name__)
        self._model = None

    def _load_model(self):
        """Lazy load YOLO model."""
        if self._model is not None:
            return

        self.logger.info(f"Loading YOLO model: {self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            ) from e

        self._model = YOLO(self.model_path)

        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self._model.to(self.device)
            self.logger.info("YOLO model loaded on CUDA")

    def detect_frames(self, frame_paths: List[Path]) -> DetectionResult:
        """
        Run detection + tracking on all frames.
        Returns DetectionResult with per-frame boxes.
        """
        self._load_model()

        boxes = []
        total_inference_time = 0.0
        self.logger.info(f"Running detection on {len(frame_paths)} frames")

        for frame_idx, frame_path in enumerate(frame_paths):
            try:
                frame = self._read_frame(frame_path)

                frame_start = time.time()
                results = self._model.track(
                    frame,
                    conf=self.confidence,
                    imgsz=self.imgsz,
                    persist=True,
                    verbose=False,
                )
                frame_time = time.time() - frame_start
                total_inference_time += frame_time

                if results and len(results) > 0:
                    result = results[0]

                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            xyxy = (
                                box.xyxy[0].cpu().numpy()
                                if hasattr(box.xyxy, "cpu")
                                else box.xyxy[0]
                            )
                            x1, y1, x2, y2 = xyxy.tolist()

                            conf = (
                                float(box.conf[0])
                                if hasattr(box.conf, "__getitem__")
                                else float(box.conf)
                            )

                            cls_idx = (
                                int(box.cls[0])
                                if hasattr(box.cls, "__getitem__")
                                else int(box.cls)
                            )
                            class_name = self._model.names[cls_idx]

                            # Get track ID if available
                            track_id = None
                            if result.boxes.id is not None:
                                track_id = int(result.boxes.id[0])

                            boxes.append(
                                DetectionBox(
                                    frame_idx=frame_idx,
                                    x1=float(x1),
                                    y1=float(y1),
                                    x2=float(x2),
                                    y2=float(y2),
                                    class_name=class_name,
                                    confidence=conf,
                                    track_id=track_id,
                                )
                            )

            except Exception as e:
                self.logger.warning(f"Error processing frame {frame_idx}: {e}")
                continue

        avg_time_ms = (total_inference_time / len(frame_paths)) * 1000 if frame_paths else 0
        self.logger.info(
            f"Detection complete: {len(boxes)} detections across {len(frame_paths)} frames "
            f"(avg {avg_time_ms:.1f}ms/frame, total {total_inference_time:.2f}s)"
        )

        return DetectionResult(
            boxes=boxes,
            fps=1.0,
            frame_count=len(frame_paths),
        )

    def _read_frame(self, frame_path: Path) -> Optional[object]:
        """Read frame using OpenCV."""
        import cv2

        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")

        return frame

    def cleanup(self):
        """Free GPU memory."""
        self.logger.info("Cleaning up YOLO detector...")
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1e9
            self.logger.info(f"VRAM after cleanup: {vram_after:.2f}GB")
