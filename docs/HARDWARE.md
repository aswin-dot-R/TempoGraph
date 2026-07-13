# Hardware Requirements — Per-Stage Breakdown

Tested on: **NVIDIA RTX 3060 6GB** + **2 × AMD GPUs** (ROCm/HIP).

All stages run on the same GPU by default. With two GPUs, vision stages
(YOLO, Depth) serialize on GPU 0 while the LLM stays warm on GPU 1.

## Stage-by-Stage Cost

| Stage | Module | Backend | VRAM | GPU | Notes |
|---|---|---|---|---|---|
| 1. Frame selection | `frame_selector.py` | CPU (numpy) | ~50 MB | — | Motion-aware sampler, no model |
| 1.5. Audio transcription | `whisper.cpp` | Vulkan / CPU | 0 (CPU) or ~300 MB (Vulkan) | NVIDIA / AMD | Vulkan uses device 1 (NVIDIA) by default to avoid `vk::DeviceLostError` on AMD gfx1201 |
| 2. YOLO detection | `detector.py` (YOLO26) | CUDA | **1–4 GB** (n/x), up to ~8 GB (l) | CUDA | `yolo26n.pt` default; auto-downloads from ultralytics/assets |
| 3. Depth estimation | `depth.py` (transformers) | CUDA | ~0.5 GB | CUDA | Optional. `Depth-Anything-V2-Small-hf`. Loaded via `transformers.pipeline` |
| 4. Frame scoring | `frame_scorer.py` | CPU (numpy) | ~50 MB | — | Top-K scorer, no model |
| 5. VLM captioning | `llama_server_backend.py` | llama-server | **~2 GB** (9B Q4), ~6 GB (35B Q4) | external | Qwen3.5-VL served by `llama.cpp` llama-server. 9B Q4_K_M is the default. Size tested: 9B, 35B |
| 6. Aggregation | `aggregator.py` | CPU (pydantic) | ~50 MB | — | Pure Python, no model |

## CPU-Only Fallback Matrix

| Configuration | What's skipped | What's kept |
|---|---|---|
| `make smoke` (default) | VLM only | YOLO n, Whisper (CPU), no depth |
| `--skip-vlm` + `--yolo-size n` | VLM, large models | YOLO n, Whisper (CPU) |
| `--skip-vlm` + `--no-depth` | VLM, depth | YOLO, Whisper (CPU) |
| `--skip-vlm` + `--no-depth` + `--no-audio` | VLM, depth, audio | YOLO only |

## Tested Hardware

| GPU | Driver | Role | Models running |
|---|---|---|---|
| NVIDIA RTX 3060 6GB | CUDA 12 | Vision (default) + Whisper Vulkan | YOLO26 n, Depth V2 Small |
| AMD 9070 XT (×2) | ROCm 6 / HIP | LLM serving | Qwen3.5-VL 9B Q4_K_M |

With a single 6 GB card, the pipeline **cannot** run the full VLM stage
(~2 GB Q4) concurrently with YOLO + Depth. The current design serializes:
YOLO/Depth → unload → VLM → unload. This is transparent to the user.

## External Services (not counted in VRAM)

| Service | Port | Notes |
|---|---|---|
| llama-server (Qwen3.5-VL) | 8085 | Started by pipeline; 9B Q4_K_M + mmproj |
| llama-server (text-only) | 8082 | Optional walker service |
| verifier | 8096 | Optional |
