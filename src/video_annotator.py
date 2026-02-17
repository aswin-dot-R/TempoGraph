import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Dict
from src.models import (
    DetectionResult,
    DetectionBox,
    DepthResult,
    AnalysisResult,
    PipelineConfig,
)


class VideoAnnotator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Color palette for different classes/tracks
        self.colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
            (200, 150, 100),
            (150, 200, 100),
            (100, 150, 200),
        ]
        self.max_color_idx = len(self.colors) - 1

    def annotate(
        self,
        video_path: str,
        output_path: str,
        detection: Optional[DetectionResult] = None,
        depth: Optional[DepthResult] = None,
        analysis: Optional[AnalysisResult] = None,
        fps: float = 1.0,
        depth_alpha: float = 0.3,
    ) -> str:
        """
        Create annotated video.

        Steps:
        1. Open original video
        2. For each frame:
           a. Draw YOLO bboxes if detection is provided
           b. Overlay depth colormap if depth is provided
           c. Draw subtitle text if analysis has audio_events
           d. Draw timestamp + active event labels at top
        3. Write to output MP4
        4. Return output path
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.logger.error(f"Error: Could not open video {video_path}")
            raise FileNotFoundError(f"Could not open video {video_path}")

        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, video_fps, (frame_width, frame_height)
        )
        if not out.isOpened():
            self.logger.error("Error: Could not open VideoWriter")
            self.cap.release()
            raise RuntimeError("Could not open VideoWriter")

        # Process frames
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Draw timestamp at top-left
            self._draw_timestamp(frame, frame_idx, video_fps)

            # Draw active events at top-right
            self._draw_active_events(frame, frame_idx, analysis)

            # Draw detection boxes if provided
            if detection and len(detection.boxes) > 0:
                self._draw_boxes(frame, detection.boxes)

            # Overlay depth map if provided
            if depth and len(depth.frames) > frame_idx:
                depth_map = np.load(depth.frames[frame_idx].depth_map_path)
                frame = self._overlay_depth(frame, depth_map, depth_alpha)

            # Write frame to output video
            out.write(frame)
            frame_idx += 1

        # Release resources
        self.cap.release()
        out.release()
        self.logger.info(f"Annotated video saved to {output_path}")
        return output_path

    def _draw_timestamp(self, frame: np.ndarray, frame_idx: int, video_fps: float):
        """Draw timestamp in MM:SS format at top-left of frame."""
        # Convert frame index to seconds
        current_time_sec = frame_idx / video_fps if video_fps > 0 else 0.0

        # Convert to MM:SS format
        minutes = int(current_time_sec // 60)
        seconds = int(current_time_sec % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"

        # Add to frame
        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    def _draw_active_events(
        self, frame: np.ndarray, frame_idx: int, analysis: Optional[AnalysisResult]
    ):
        """Draw active event labels at top-right of frame."""
        if not analysis:
            return

        # Get current time in seconds
        current_time_sec = (
            frame_idx / self.cap.get(cv2.CAP_PROP_FPS)
            if hasattr(self.cap, "get")
            else 0.0
        )

        # Find active events at this timestamp
        active_event_labels = self._get_active_subtitle(analysis, current_time_sec)

        if active_event_labels:
            # Position at top-right
            text_position = (frame.shape[1] - 200, 30)
            self._draw_text_with_background(frame, active_event_labels, text_position)

    def _get_active_subtitle(
        self, analysis: AnalysisResult, current_time_sec: float
    ) -> Optional[str]:
        """Get subtitle text active at current timestamp."""
        labels = []

        # Check visual events
        for event in analysis.visual_events:
            # Simple overlap check - could be improved
            start_sec = self._time_to_seconds(event.start_time)
            end_sec = self._time_to_seconds(event.end_time)
            if start_sec <= current_time_sec <= end_sec:
                labels.append(event.description)

        # Check audio events
        for audio_event in analysis.audio_events:
            start_sec = self._time_to_seconds(audio_event.start_time)
            end_sec = self._time_to_seconds(audio_event.end_time)
            if start_sec <= current_time_sec <= end_sec:
                if audio_event.text:
                    labels.append(f"speech: {audio_event.text}")

        if labels:
            return "; ".join(labels)
        return None

    def _draw_text_with_background(self, frame: np.ndarray, text: str, position: tuple):
        """Draw text with semi-transparent background."""
        x, y = position

        # Create a rectangle background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - 30), (x + 200, y + 30), (0, 0, 0), cv2.FILLED)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Put text
        cv2.putText(
            frame,
            text,
            (x + 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def _draw_boxes(self, frame: np.ndarray, boxes: List[DetectionBox]):
        """Draw bounding boxes with labels on frame."""
        for i, box in enumerate(boxes):
            if i >= len(self.colors):
                i = 0
            color = self.colors[i % len(self.colors)]

            # Draw rectangle
            cv2.rectangle(
                frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2
            )

            # Put label
            label = f"{box.class_name} {box.track_id or 0} ({box.confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (int(box.x1), int(box.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    def _overlay_depth(
        self, frame: np.ndarray, depth_map: np.ndarray, alpha: float = 0.3
    ) -> np.ndarray:
        """Overlay depth map as semi-transparent colormap."""
        # Normalize depth map to 0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Apply colormap
        colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

        # Resize to match frame if needed
        if colored_depth.shape[:2] != frame.shape[:2]:
            colored_depth = cv2.resize(colored_depth, (frame.shape[1], frame.shape[0]))

        # Overlay with alpha blending
        return cv2.addWeighted(frame, 1 - alpha, colored_depth, alpha, 0)

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS format to seconds."""
        if ":" in time_str:
            parts = time_str.split(":")
            return int(parts[0]) * 60 + float(parts[1])
        return float(time_str)

    def _get_active_events(
        self, analysis: AnalysisResult, current_time_sec: float
    ) -> List[str]:
        """Get list of active event descriptions at current time."""
        events = []
        # This is a simplified version - could be enhanced
        return events

    def cleanup(self):
        """Remove extracted frames and audio."""
        import shutil

        # Note: VideoAnnotator doesn't directly manage frame extraction artifacts
        # This method exists for interface consistency
        pass
