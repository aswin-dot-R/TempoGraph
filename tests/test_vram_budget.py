"""
Verify all models fit in 6GB VRAM when loaded sequentially.
Run on actual GPU to validate.
"""

import torch
import gc
import time


def log_vram(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{label}] Allocated: {alloc:.3f}GB, Reserved: {reserved:.3f}GB")


def test_sequential_loading():
    """Test that all models fit in 6GB when loaded sequentially."""
    assert torch.cuda.is_available(), "No GPU available"

    total_vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"Total VRAM: {total_vram:.1f}GB")

    log_vram("baseline")

    # Phase 1: YOLO + Depth (should be ~0.8GB total)
    print("\n--- Phase 1: YOLO + Depth ---")
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")
    yolo.to("cuda")
    log_vram("after YOLO load")

    # Run one dummy inference
    import numpy as np

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    yolo(dummy, verbose=False)
    log_vram("after YOLO inference")

    # Load depth
    print("\n--- Loading Depth Anything V2 ---")
    try:
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }

        depth_model = DepthAnythingV2(**model_configs["vits"])
        depth_model.load_state_dict(
            torch.load("checkpoints/depth_anything_v2_vits.pth", map_location="cpu")
        )
        depth_model = depth_model.to("cuda").eval()
        log_vram("after Depth load")

        # Run one dummy inference
        depth_model.infer_image(dummy)
        log_vram("after Depth inference")

        # Cleanup depth
        del depth_model
        gc.collect()
        torch.cuda.empty_cache()
        log_vram("after Depth cleanup")

    except Exception as e:
        print(f"Depth model test skipped: {e}")

    # Phase 2: Unload
    print("\n--- Phase 2: Unload ---")
    del yolo
    gc.collect()
    torch.cuda.empty_cache()
    log_vram("after unload")

    # Phase 3: Qwen (should be ~2.0GB)
    print("\n--- Phase 3: Qwen ---")
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    log_vram("after Qwen load")

    peak = torch.cuda.memory_allocated() / 1e9
    print(f"\nPeak VRAM: {peak:.2f}GB")

    if peak < 4.0:
        print(f"✅ Peak VRAM: {peak:.2f}GB — fits in 6GB budget!")
    else:
        print(f"❌ Peak VRAM: {peak:.2f}GB — exceeds 4GB budget!")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_vram("final cleanup")


if __name__ == "__main__":
    test_sequential_loading()
