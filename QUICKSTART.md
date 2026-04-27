# TempoGraph v2 — Quickstart

## One-line install

```bash
git clone https://github.com/aswin-dot-R/TempoGraph.git && cd TempoGraph && make install
```

What that does:
1. Detects your Python env (uses an active conda env, an existing
   `.venv/`, or creates a fresh `.venv/`)
2. Installs Python deps via `pip install -r requirements.txt`
3. Clones `whisper.cpp` to `~/whisper.cpp` and builds it with **Vulkan**
   (falls back to CUDA → CPU based on what's installed)
4. Downloads the `base.en` Whisper model (~141 MB)
5. Creates `.streamlit/config.toml` with a 4 GB upload cap

You can re-run `make install` any time — every step is idempotent.

## Run the UI

```bash
make run
```

Opens at `http://localhost:8501`. Drag a video into the upload box,
tweak the sidebar, click **Run full pipeline**.

## Run on a single video from the CLI

```bash
make run-cli VIDEO=path/to/clip.mp4
```

Outputs to `results/<basename>/`.

## Want the local LLM too? (~10 GB)

The default install **skips** the Qwen3.5-VL setup because the model
weights are large. Without the LLM, you can still run YOLO + Depth +
Audio + Frame selection — only the captioning stage requires a VLM.

To get the full local stack:

```bash
make install-llm
```

What that adds:
- Clones `llama.cpp` to `~/llama.cpp` and builds it with HIP/CUDA/Vulkan
  (whichever is available)
- Downloads `Qwen3.5-9B-Q8_0.gguf` + the BF16 vision projector from
  HuggingFace (~10 GB) into `~/qwen-models/`
- Generates a systemd `--user` unit `qwen-tempograph.service` pointing
  at those paths

Enable + start the service:
```bash
systemctl --user enable --now qwen-tempograph.service
curl -s http://127.0.0.1:8082/v1/models   # confirm it's up
```

Then re-launch the UI — it'll auto-start/stop the service per run when
you tick **Keep VLM running after this video** off.

## Test it works

```bash
make test          # 29 unit tests (~3 s)
make smoke         # synthetic 10s video, runs the full pipeline end-to-end with --skip-vlm
```

## Manual install (if `make install` fails)

```bash
# 1. Python env (use any of these)
python3 -m venv .venv && source .venv/bin/activate
# OR: conda create -n tempograph python=3.12 && conda activate tempograph

# 2. Python deps
pip install -r requirements.txt

# 3. whisper.cpp
git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_VULKAN=1   # or -DGGML_CUDA=1 if you have the CUDA toolkit
cmake --build build -j --config Release
bash models/download-ggml-model.sh base.en
cd -

# 4. (optional) llama.cpp + Qwen
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_VULKAN=1   # or -DGGML_HIP=1 / -DGGML_CUDA=1
cmake --build build -j --target llama-server --config Release
huggingface-cli download lmstudio-community/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-Q8_0.gguf mmproj-Qwen3.5-9B-BF16.gguf \
  --local-dir ~/qwen-models --local-dir-use-symlinks False
cd -
```

## Troubleshooting

**`make install` fails on Vulkan**: install Vulkan loader + headers:
```
sudo apt install vulkan-tools libvulkan-dev mesa-vulkan-drivers
```

**`whisper-cli` segfaults on AMD GPU**: that's `radv vk::DeviceLostError` —
the pipeline auto-falls-back to NVIDIA → CPU when this happens, but you
can also force CPU per-run with `--whisper-gpu-device -1` (sidebar
"CPU only").

**Qwen download is slow**: it's a 10 GB model. The `huggingface-cli`
download supports resume with `--resume-download`.

**Port 8082 in use**: another LLM server is on it. Either stop the
other service or change `--vlm-url` to a different port and update the
systemd unit's `--port` flag.

**Streamlit refuses to upload large files**: the bootstrap creates
`.streamlit/config.toml` with a 4 GB cap. If you blew it away, recreate
or pass `--server.maxUploadSize=4096` on launch.

## What's next

See [`README.md`](README.md) for the architecture overview and
[`docs/PIPELINE.md`](docs/PIPELINE.md) for stage-by-stage internals.
