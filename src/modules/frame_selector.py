"""Two-mode frame selection for TempoGraph v2.

Static mode: pixel-delta keyframes + uniform sampling.
Moving mode: motion-compensated residual delta after homography warp.
Auto: detect mode from first 30 sampled frames' ORB displacement.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from src.models import CameraMode, FrameSelectionResult


class FrameSelector:
    def __init__(self, thumb_width: int = 160, orb_features: int = 500):
        self.thumb_width = thumb_width
        self.orb_features = orb_features
        self.logger = logging.getLogger(__name__)

    def select(
        self,
        video_path: str,
        camera_mode: CameraMode = CameraMode.STATIC,
        sample_fps: float = 1.0,
        threshold_mult: float = 1.0,
    ) -> FrameSelectionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Scan at up to 10 fps regardless of sample_fps, to catch fast transitions.
        # sample_fps only governs the uniform-sampling grid, not the change-detection scan.
        scan_fps = max(sample_fps * 2, min(10.0, video_fps))
        scan_interval = max(1, int(round(video_fps / scan_fps)))

        if camera_mode == CameraMode.AUTO:
            camera_mode = self._auto_detect_mode(cap, scan_interval, width, height)

        if camera_mode == CameraMode.STATIC:
            scan_indices, deltas = self._scan_pixel_deltas(
                cap, scan_interval, width, height
            )
        else:
            scan_indices, deltas = self._scan_motion_compensated_deltas(
                cap, scan_interval, width, height
            )

        threshold = self._compute_threshold(deltas, threshold_mult)
        keyframe_indices = self._extract_keyframes(scan_indices, deltas, threshold)
        sampled_indices = self._uniform_sample(total, video_fps, sample_fps)

        union = sorted(set(keyframe_indices) | set(sampled_indices))

        cap.release()

        self.logger.info(
            f"Frame selection ({camera_mode.value}): {len(union)} frames "
            f"({len(keyframe_indices)} keyframe + {len(sampled_indices)} sampled), "
            f"threshold={threshold:.2f}"
        )

        return FrameSelectionResult(
            frame_indices=union,
            keyframe_indices=keyframe_indices,
            sampled_indices=sampled_indices,
            scan_indices=scan_indices,
            deltas=deltas,
            threshold=threshold,
            camera_mode=camera_mode,
        )

    def _scan_pixel_deltas(
        self, cap: cv2.VideoCapture, scan_interval: int, width: int, height: int
    ) -> Tuple[List[int], List[float]]:
        thumb_w = min(self.thumb_width, width)
        thumb_h = max(1, int(height * thumb_w / width)) if width > 0 else 120

        indices: List[int] = []
        deltas: List[float] = []
        prev = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % scan_interval == 0:
                small = cv2.resize(frame, (thumb_w, thumb_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev is None:
                    deltas.append(0.0)
                else:
                    deltas.append(float(np.mean(np.abs(gray - prev))))
                indices.append(i)
                prev = gray
            i += 1
        return indices, deltas

    def _scan_motion_compensated_deltas(
        self, cap: cv2.VideoCapture, scan_interval: int, width: int, height: int
    ) -> Tuple[List[int], List[float]]:
        thumb_w = min(self.thumb_width, width)
        thumb_h = max(1, int(height * thumb_w / width)) if width > 0 else 120
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        indices: List[int] = []
        deltas: List[float] = []
        prev_small = None
        prev_gray = None
        prev_kp = None
        prev_des = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % scan_interval == 0:
                small = cv2.resize(frame, (thumb_w, thumb_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)

                if prev_gray is None or des is None or prev_des is None:
                    deltas.append(0.0)
                else:
                    delta = self._residual_after_warp(
                        prev_gray, gray, prev_kp, kp, prev_des, des, bf
                    )
                    deltas.append(delta)
                indices.append(i)
                prev_gray = gray
                prev_kp = kp
                prev_des = des
            i += 1
        return indices, deltas

    def _residual_after_warp(
        self, prev_gray, gray, prev_kp, kp, prev_des, des, bf
    ) -> float:
        try:
            matches = bf.match(prev_des, des)
            if len(matches) < 8:
                return float(np.mean(np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32))))

            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if H is None:
                return float(np.mean(np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32))))

            h, w = gray.shape
            warped = cv2.warpPerspective(prev_gray, H, (w, h))
            return float(np.mean(np.abs(gray.astype(np.float32) - warped.astype(np.float32))))
        except cv2.error:
            return float(np.mean(np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32))))

    def _compute_threshold(self, deltas: List[float], mult: float) -> float:
        non_zero = [d for d in deltas if d > 0]
        if not non_zero:
            return 0.0
        arr = np.array(non_zero)
        return float(np.median(arr) + mult * np.std(arr))

    def _extract_keyframes(
        self, indices: List[int], deltas: List[float], threshold: float
    ) -> List[int]:
        kf = [indices[i] for i, d in enumerate(deltas) if d >= threshold and threshold > 0]
        # Always include first frame
        if indices and indices[0] not in kf:
            kf.insert(0, indices[0])
        return sorted(set(kf))

    def _uniform_sample(self, total_frames: int, video_fps: float, sample_fps: float) -> List[int]:
        if sample_fps <= 0 or video_fps <= 0:
            return []
        step = max(1, int(round(video_fps / sample_fps)))
        return list(range(0, total_frames, step))

    def _auto_detect_mode(
        self, cap: cv2.VideoCapture, scan_interval: int, width: int, height: int
    ) -> CameraMode:
        """Estimate dominant ORB displacement over first 30 sampled frames."""
        thumb_w = min(self.thumb_width, width)
        thumb_h = max(1, int(height * thumb_w / width))
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_kp = None
        prev_des = None
        displacements: List[float] = []
        i = 0
        sampled = 0
        while sampled < 30:
            ret, frame = cap.read()
            if not ret:
                break
            if i % scan_interval == 0:
                small = cv2.resize(frame, (thumb_w, thumb_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)
                if prev_des is not None and des is not None and len(kp) > 0 and len(prev_kp) > 0:
                    matches = bf.match(prev_des, des)
                    if matches:
                        d = np.mean(
                            [np.linalg.norm(np.array(prev_kp[m.queryIdx].pt) - np.array(kp[m.trainIdx].pt)) for m in matches]
                        )
                        displacements.append(d)
                prev_kp = kp
                prev_des = des
                sampled += 1
            i += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind for caller

        if not displacements:
            return CameraMode.STATIC
        median_disp = float(np.median(displacements))
        threshold_disp = 0.05 * thumb_w
        return CameraMode.MOVING if median_disp > threshold_disp else CameraMode.STATIC
