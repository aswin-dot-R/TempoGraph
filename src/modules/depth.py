"""
Depth Anything V2 Small module.

Model: depth-anything-v2-vits (~98MB weights, ViT-Small encoder)
VRAM: ~0.5GB when loaded
Speed: ~100ms per frame on GPU

EXACT USAGE:
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,
             'out_channels': [48, 96, 192, 384]},
}

model = DepthAnythingV2(**model_configs['vits'])
model.load_state_dict(
    torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu')
)
model = model.to('cuda').eval()

# Inference:
raw_img = cv2.imread(str(frame_path))
depth = model.infer_image(raw_img)  # HxW numpy array, float32
# depth values: relative (not metric), higher = farther

# Colorize for visualization:
depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255
depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

WEIGHT DOWNLOAD:
The model weights need to be downloaded separately.
Check if file exists, if not, download from HuggingFace:
huggingface_hub.hf_hub_download(
    repo_id="depth-anything/Depth-Anything-V2-Small",
    filename="depth_anything_v2_vits.pth",
    local_dir="checkpoints"
)

ALTERNATIVE: Use transformers pipeline (simpler but slightly more VRAM):
from transformers import pipeline
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small")
result = pipe(image)  # returns {"depth": PIL.Image, "predicted_depth": tensor}

VRAM management: same pattern — load, process all frames, cleanup.
"""

import os
import gc
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List
from src.models import DepthResult, DepthFrame

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class DepthEstimator:
    def __init__(self, model_variant: str = "vits", device: str = "cuda"):
        self.model_variant = model_variant
        self.device = device
        self.logger = logging.getLogger(__name__)
        self._model = None

    def _load_model(self):
        """Lazy load Depth Anything V2 model."""
        if self._model is not None:
            return

        self.logger.info(f"Loading Depth Anything V2 ({self.model_variant})")

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            raise ImportError(
                "depth_anything_v2 not installed. "
                "Install with: pip install depth-anything-v2"
            ) from e

        # Model config
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }

        # Load model
        self._model = DepthAnythingV2(**model_configs[self.model_variant])

        # Load weights
        weights_path = self._get_weights_path()
        self._model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

        # Move to device
        self._model = self._model.to(self.device).eval()

        self.logger.info("Depth model loaded successfully")

    def _get_weights_path(self) -> str:
        """Get path to model weights."""
        # Try common locations
        possible_paths = [
            "checkpoints/depth_anything_v2_vits.pth",
            "depth_anything_v2_vits.pth",
            os.path.join(
                os.path.dirname(__file__), "checkpoints", "depth_anything_v2_vits.pth"
            ),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Try to download from HuggingFace
        self.logger.warning("Weights not found. Attempting to download...")
        try:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(
                repo_id="depth-anything/Depth-Anything-V2-Small",
                filename="depth_anything_v2_vits.pth",
                local_dir="checkpoints",
            )
            self.logger.info(f"Weights downloaded to: {weights_path}")
            return weights_path
        except Exception as e:
            raise FileNotFoundError(
                f"Depth model weights not found at any location. "
                f"Please download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Small\n"
                f"Error: {e}"
            ) from e

    def estimate_frames(
        self, frame_paths: List[Path], output_dir: str = "/tmp/tempograph_depth"
    ) -> DepthResult:
        """
        Run depth estimation on all frames.
        Saves depth maps as .npy and colorized .png files.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        self._load_model()

        frames = []

        self.logger.info(f"Running depth estimation on {len(frame_paths)} frames")

        for frame_idx, frame_path in enumerate(frame_paths):
            try:
                # Read frame
                raw_img = cv2.imread(str(frame_path))
                if raw_img is None:
                    self.logger.warning(f"Failed to read frame: {frame_path}")
                    continue

                # Run inference
                depth = self._model.infer_image(raw_img)

                # Normalize depth for visualization
                if depth.max() > depth.min():
                    depth_norm = (
                        (depth - depth.min()) / (depth.max() - depth.min()) * 255
                    )
                else:
                    depth_norm = np.zeros_like(depth)
                depth_norm = depth_norm.astype(np.uint8)

                # Colorize
                depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

                # Save depth map
                depth_path = os.path.join(output_dir, f"depth_{frame_idx:06d}.npy")
                np.save(depth_path, depth)

                # Save colorized depth
                color_path = os.path.join(output_dir, f"depth_{frame_idx:06d}.png")
                cv2.imwrite(color_path, depth_colored)

                frames.append(
                    DepthFrame(frame_idx=frame_idx, depth_map_path=depth_path)
                )

                if (frame_idx + 1) % 10 == 0:
                    self.logger.info(
                        f"Processed {frame_idx + 1}/{len(frame_paths)} frames"
                    )

            except Exception as e:
                self.logger.warning(f"Error processing frame {frame_idx}: {e}")
                continue

        self.logger.info(f"Depth estimation complete: {len(frames)} frames")

        return DepthResult(frames=frames)

    def estimate_to_db(
        self,
        frame_indices,
        frame_paths,
        db,
        output_dir: str,
    ) -> None:
        """Run depth estimation; save .npy per frame; populate per-bbox mean depth."""
        os.makedirs(output_dir, exist_ok=True)
        self._load_model()
        self.logger.info(f"Depth-to-DB: {len(frame_paths)} frames")

        for frame_idx, frame_path in zip(frame_indices, frame_paths):
            try:
                raw = cv2.imread(str(frame_path))
                if raw is None:
                    self.logger.warning(f"Could not read {frame_path}")
                    continue
                depth = self._model.infer_image(raw)

                # Normalize to [0,1] and save as .npy
                d_min, d_max = float(depth.min()), float(depth.max())
                if d_max > d_min:
                    depth_norm = (depth - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(depth)

                npy_path = os.path.join(output_dir, f"depth_{frame_idx:06d}.npy")
                np.save(npy_path, depth_norm.astype(np.float32))
                db.insert_depth_frame(frame_idx=int(frame_idx), depth_npy_path=npy_path)

                # Per-bbox mean depth computed from raw depth values
                h, w = depth.shape[:2]
                for det in db.get_detections_for_frame(int(frame_idx)):
                    bx1 = max(0, int(det["x1"] * w))
                    by1 = max(0, int(det["y1"] * h))
                    bx2 = min(w, int(det["x2"] * w))
                    by2 = min(h, int(det["y2"] * h))
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    region = depth[by1:by2, bx1:bx2]
                    if region.size == 0:
                        continue
                    mean_d = float(np.mean(region))
                    db.set_detection_mean_depth(det["detection_id"], mean_d)
            except Exception as e:
                self.logger.warning(f"depth-to-db error frame {frame_idx}: {e}")
                continue

    def cleanup(self):
        """Free GPU memory."""
        self.logger.info("Cleaning up depth estimator...")
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if self.device == "cuda" and _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1e9
            self.logger.info(f"VRAM after cleanup: {vram_after:.2f}GB")
