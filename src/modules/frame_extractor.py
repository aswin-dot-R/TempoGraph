import cv2
import subprocess
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    frame_paths: List[Path]
    audio_path: Optional[Path]
    total_frames: int
    video_fps: float
    video_duration: float  # seconds
    extraction_fps: float  # what we sampled at
    width: int
    height: int


class FrameExtractor:
    def __init__(self, output_dir: str = "/tmp/tempograph_frames"):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

    def extract(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 60,
        resize_width: Optional[int] = 640,
        extract_audio: bool = True,
    ) -> ExtractionResult:
        """
        Extract frames using adaptive keyframe detection based on pixel deltas.

        Instead of sampling at a constant rate, we:
        1. Scan the video at a coarse interval computing frame-to-frame deltas
        2. Identify keyframes where delta exceeds a threshold (scene changes)
        3. Between keyframes, select fewer frames proportional to segment stability
        4. This yields dense sampling around visual changes, sparse in static segments
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for f in self.output_dir.glob("frame_*.jpg"):
            f.unlink()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / video_fps if video_fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(
            f"Video: {video_fps:.1f} FPS, {frame_count} frames, "
            f"{width}x{height}, {video_duration:.1f}s"
        )

        # Pass 1: scan deltas at a coarse interval to find keyframes
        scan_interval = max(1, int(video_fps / max(fps * 2, 2.0)))
        candidate_indices, deltas = self._scan_deltas(cap, scan_interval, width, height)

        if not candidate_indices:
            cap.release()
            raise RuntimeError("No frames could be read from video")

        # Pass 2: select frames using delta-weighted keyframe logic
        selected_indices = self._select_keyframes(
            candidate_indices, deltas, max_frames
        )

        self.logger.info(
            f"Keyframe selection: {len(candidate_indices)} candidates -> "
            f"{len(selected_indices)} selected"
        )

        # Pass 3: seek and save the selected frames
        frame_paths = self._save_frames(
            cap, selected_indices, resize_width, width, height
        )

        cap.release()

        if not frame_paths:
            raise RuntimeError("No frames extracted")

        # Audio extraction
        audio_path = None
        if extract_audio:
            audio_path = self._extract_audio(video_path)

        extraction_fps = len(frame_paths) / video_duration if video_duration > 0 else fps

        self.logger.info(
            f"Extracted {len(frame_paths)} frames (effective {extraction_fps:.2f} FPS), "
            f"audio={'yes' if audio_path else 'no'}"
        )

        return ExtractionResult(
            frame_paths=frame_paths,
            audio_path=audio_path,
            total_frames=frame_count,
            video_fps=video_fps,
            video_duration=video_duration,
            extraction_fps=extraction_fps,
            width=width,
            height=height,
        )

    def _scan_deltas(
        self,
        cap: cv2.VideoCapture,
        scan_interval: int,
        width: int,
        height: int,
    ) -> Tuple[List[int], List[float]]:
        """
        First pass: read frames at scan_interval, compute mean absolute pixel
        delta between consecutive frames (on downscaled grayscale).
        Returns (frame_indices, deltas) where deltas[i] is the change from
        frame i-1 to frame i (deltas[0] = max delta so the first frame is
        always treated as a keyframe).
        """
        # Downscale for fast delta computation
        thumb_w = min(160, width)
        thumb_h = int(height * thumb_w / width) if width > 0 else 120

        candidate_indices: List[int] = []
        deltas: List[float] = []
        prev_gray = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % scan_interval == 0:
                small = cv2.resize(frame, (thumb_w, thumb_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

                if prev_gray is None:
                    # First frame always gets max delta so it's always selected
                    deltas.append(float("inf"))
                else:
                    delta = np.mean(np.abs(gray - prev_gray))
                    deltas.append(float(delta))

                candidate_indices.append(frame_idx)
                prev_gray = gray

            frame_idx += 1

        return candidate_indices, deltas

    def _select_keyframes(
        self,
        indices: List[int],
        deltas: List[float],
        max_frames: int,
    ) -> List[int]:
        """
        Select up to max_frames from candidates using H.264-style keyframe logic:

        1. Compute an adaptive threshold from the delta distribution
        2. Mark frames above threshold as keyframes (I-frames)
        3. Keyframes are always selected
        4. Between keyframes, distribute remaining budget proportional to
           the cumulative delta of each segment (more change = more frames)
        5. Within each segment, pick frames with the highest deltas
        """
        n = len(indices)
        if n <= max_frames:
            return list(indices)

        # Replace inf with a value above all real deltas for sorting
        real_deltas = [d for d in deltas if d != float("inf")]
        max_real = max(real_deltas) if real_deltas else 1.0
        working_deltas = [d if d != float("inf") else max_real * 2 for d in deltas]

        # Adaptive threshold: median + 1 stddev of non-inf deltas
        if real_deltas:
            arr = np.array(real_deltas)
            threshold = float(np.median(arr) + np.std(arr))
        else:
            threshold = 0.0

        # Identify keyframe positions (always include first and last)
        keyframe_mask = [False] * n
        keyframe_mask[0] = True
        keyframe_mask[-1] = True
        for i, d in enumerate(working_deltas):
            if d >= threshold:
                keyframe_mask[i] = True

        keyframe_positions = [i for i, is_kf in enumerate(keyframe_mask) if is_kf]
        num_keyframes = len(keyframe_positions)

        if num_keyframes >= max_frames:
            # Too many keyframes — pick the top max_frames by delta
            scored = sorted(
                range(n), key=lambda i: working_deltas[i], reverse=True
            )
            selected_pos = sorted(scored[:max_frames])
            return [indices[i] for i in selected_pos]

        # Budget for inter-keyframe fills
        fill_budget = max_frames - num_keyframes

        # Build segments between consecutive keyframes
        segments = []
        for seg_idx in range(len(keyframe_positions) - 1):
            start = keyframe_positions[seg_idx]
            end = keyframe_positions[seg_idx + 1]
            # Non-keyframe frames in this segment
            between = [
                i for i in range(start + 1, end) if not keyframe_mask[i]
            ]
            seg_delta_sum = sum(working_deltas[i] for i in between) if between else 0.0
            segments.append((between, seg_delta_sum))

        # Distribute fill budget proportional to cumulative delta per segment
        total_seg_delta = sum(s[1] for s in segments)
        selected_set = set(keyframe_positions)

        if total_seg_delta > 0:
            for between, seg_delta in segments:
                if not between:
                    continue
                # Proportional allocation (at least 0)
                seg_alloc = max(0, round(fill_budget * seg_delta / total_seg_delta))
                seg_alloc = min(seg_alloc, len(between))
                # Pick the top seg_alloc frames by delta within the segment
                top = sorted(between, key=lambda i: working_deltas[i], reverse=True)
                for i in top[:seg_alloc]:
                    selected_set.add(i)
        else:
            # All deltas are zero (static video) — spread evenly
            non_kf = [i for i in range(n) if not keyframe_mask[i]]
            if non_kf and fill_budget > 0:
                step = max(1, len(non_kf) // fill_budget)
                for i in range(0, len(non_kf), step):
                    if len(selected_set) >= max_frames:
                        break
                    selected_set.add(non_kf[i])

        # Trim to max_frames if rounding overshot
        selected_sorted = sorted(selected_set)
        if len(selected_sorted) > max_frames:
            # Keep keyframes, trim lowest-delta fills
            fills = [i for i in selected_sorted if not keyframe_mask[i]]
            fills.sort(key=lambda i: working_deltas[i])
            excess = len(selected_sorted) - max_frames
            remove = set(fills[:excess])
            selected_sorted = [i for i in selected_sorted if i not in remove]

        return [indices[i] for i in selected_sorted]

    def _save_frames(
        self,
        cap: cv2.VideoCapture,
        selected_indices: List[int],
        resize_width: Optional[int],
        width: int,
        height: int,
    ) -> List[Path]:
        """Seek to each selected frame index and save as JPEG."""
        frame_paths = []
        for frame_idx in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if resize_width and width > 0:
                scale = resize_width / width
                new_h = int(height * scale)
                frame = cv2.resize(frame, (resize_width, new_h))

            path = self.output_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(path), frame)
            frame_paths.append(path)

        return frame_paths

    def _extract_audio(self, video_path: str) -> Optional[Path]:
        """Extract audio as 16kHz mono WAV using ffmpeg."""
        try:
            audio_path = self.output_dir / "audio.wav"
            result = subprocess.run(
                [
                    "ffmpeg", "-i", video_path, "-vn",
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(audio_path), "-y",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.logger.warning(f"Audio extraction failed: {result.stderr}")
                return None
            self.logger.info("Audio extracted successfully")
            return audio_path
        except FileNotFoundError:
            self.logger.warning("ffmpeg not found, skipping audio extraction")
            return None
        except Exception as e:
            self.logger.warning(f"Audio extraction error: {e}")
            return None

    def cleanup(self):
        """Remove extracted frames and audio."""
        import shutil

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
