# TempoGraph

**Turn video into structured entities, events, captions, transcripts, and graphs.**

TempoGraph is a fully-local multimodal video-analysis pipeline. YOLO + Depth
Anything V2 + Whisper.cpp + Qwen3.5-VL, glued together by a chunked
orchestrator that persists everything to a per-run SQLite store, with a
Streamlit UI for both kicking off runs and browsing past results.

> **Quick install: see [`QUICKSTART.md`](QUICKSTART.md).**
> **Deep technical doc: see [`docs/PIPELINE.md`](docs/PIPELINE.md)** — every
> stage, every knob, every output file, every API call we make. This README
> is the architecture overview; that doc is the source of truth.

---

## One-line install

```bash
git clone https://github.com/aswin-dot-R/TempoGraph.git && cd TempoGraph && make install
```

What happens: detects your Python env, installs deps, clones + builds
`whisper.cpp` (Vulkan / CUDA / CPU auto-detected), downloads the
`base.en` whisper model, and writes `.streamlit/config.toml` with a
4 GB upload cap. Idempotent — safe to re-run.

For the local LLM (~10 GB more): `make install-llm`. See
[`QUICKSTART.md`](QUICKSTART.md) for details.

---

## Hardware split (recommended)

The pipeline is designed around two discrete GPUs but works fine with one:

| GPU | Role | Used by |
|---|---|---|
| NVIDIA / CUDA + Vulkan | Vision + ASR | YOLO, Depth Anything V2, Whisper.cpp |
| AMD ROCm/HIP **or** any Vulkan GPU | LLM serving | llama.cpp / Qwen3.5-VL captioning |

The split is intentional — the LLM-serving GPU loads the 9 B Qwen
weights once and stays warm; the vision GPU handles all the torch +
Vulkan compute that runs sequentially per-frame. With a single GPU, the
pipeline serializes — qwen autostarts before VLM stage and autostops
after, so it doesn't fight with YOLO/Depth for VRAM.

---

## Run it

```bash
make run        # Streamlit UI on http://localhost:8501
```

Drag a video in, tweak the sidebar, click **Run full pipeline**. Every
external service (qwen LLM, Whisper, model weights) is pulled / started
on demand.

CLI equivalent:
```bash
make run-cli VIDEO=path/to/clip.mp4
```

---

## V2 pipeline — what runs end to end

Stages execute strictly in order. Each persists its output to
`results/<video_filename>/tempograph.db` (a per-run SQLite store) and emits a
stage event the UI renders live.

| # | Stage | Code path | What it produces |
|---|---|---|---|
| 1 | **Frame selection** | `src/modules/frame_selector.py` | motion-aware sampled & keyframe indices |
| 1.5 | **Audio transcription** *(opt-in)* | `src/modules/whisper_cpp.py` | `audio_segments` rows + `transcript.json` |
| 2 | **YOLO detection** | `src/modules/detector.py` | `detections` rows |
| 3 | **Depth estimation** *(opt-in)* | `src/modules/depth.py` | `depth_frames` rows + per-bbox `mean_depth` |
| 4 | **Frame scoring** | `src/modules/frame_scorer.py` | top-K frames for the VLM |
| 5 | **VLM captioning** | `src/backends/llama_server_backend.py` | `chunks.json` (per-chunk Qwen output) |
| 6 | **Aggregation** | `src/aggregator.py` | `analysis.json` (entities, visual_events, audio_events, multimodal_correlations) |

A `VLM autostart` step appears between Stage 4 and Stage 5 if the qwen
service isn't already running, and a matching `VLM autostop` runs after
Stage 6 if "Keep VLM running" is unchecked.

### Per-run output directory

Every run writes everything under `results/<filename>/`:

```
results/my_clip.mp4/
├── tempograph.db        ← single SQLite, all stages
├── frames/              ← saved JPEGs (downscaled to frame_max_width=640)
│   └── frame_000000.jpg ...
├── depth/               ← .npy depth maps (only when --depth)
├── transcript.json      ← whisper segments (only when --audio)
├── chunks.json          ← raw per-chunk Qwen outputs
├── analysis.json        ← final structured analysis
├── annotated_strip.mp4  ← optional, built on demand from the Results UI
└── graph.html           ← pyvis graph (when pyvis is installed)
```

### SQLite schema

```sql
frames(frame_idx PK, timestamp_ms, image_path, is_keyframe, delta_score)
detections(detection_id PK, frame_idx FK, track_id, class_name,
           x1, y1, x2, y2, confidence, mean_depth)
depth_frames(frame_idx PK, depth_npy_path)
audio_segments(segment_id PK, start_ms, end_ms, text,
               no_speech_prob, avg_logprob)
```

Bbox coords (`x1..y2`) are **normalised to the saved JPEG dimensions**, not
the source video — so `x_pixel = x_norm × jpeg_width`. (The original code
had a scale-factor bug here that was fixed; old runs from before that fix
have wrong bboxes.)

---

## External services it depends on

### Qwen3.5-VL via llama.cpp llama-server

- systemd `--user` unit: `gemma4-turboquant.service` (text-only, port 8081),
  `qwen35-turboquant.service` (vision, port 8082)
- Binary: `/home/ashie/llama-cpp-turboquant/build/bin/llama-server`
  (built with `libggml-hip.so` → runs on AMD 9070 XT)
- Model: `/home/ashie/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf`
  + `mmproj-Qwen3.5-9B-BF16.gguf` for vision
- Context window: 100 096 tokens (1 lakh) with `--cache-type-k turbo3 --cache-type-v turbo3`
- VRAM: ~11 GiB during inference

Pipeline auto-starts this service via `systemctl --user start
qwen35-turboquant.service` if it isn't already up, and stops it after the
run unless **Keep VLM running** is checked. Important quirks the backend
worked around:

- llama-server uses OpenAI-style `/v1/chat/completions`, *not* Ollama's
  `/api/chat`. Image content is sent as `image_url` data URIs, not as the
  Ollama-native `images` array.
- Qwen3 has a hidden reasoning channel that eats `max_tokens` if not
  disabled. Every request passes
  `chat_template_kwargs: {enable_thinking: false}`. The prompt also
  prepends `/no_think` as a belt-and-suspenders.

### Whisper.cpp

- Built from `https://github.com/ggml-org/whisper.cpp` at `/home/ashie/whisper.cpp`
- Backend: **Vulkan** (`-DGGML_VULKAN=1`), runs on the **NVIDIA 3060** by
  default (Vulkan device 1). Device 0 is AMD radv which sometimes hits
  `vk::DeviceLostError` on gfx1201.
- Models live in `/home/ashie/whisper.cpp/models/ggml-<name>.bin`. The
  pipeline's `WhisperCppTranscriber.ensure_model_downloaded()` invokes
  `download-ggml-model.sh <name>` if a model is missing.
- Available models: `tiny / tiny.en / base / base.en / small / small.en /
  medium / medium.en / large-v1 / large-v2 / large-v3 / large-v3-turbo`.

### Depth Anything V2

Loaded via the `transformers` pipeline (`pipeline("depth-estimation",
"depth-anything/Depth-Anything-V2-Small-hf")`). Weights auto-download from
HF on first use. The `depth-anything-v2` PyPI package is unsatisfiable
(only `0.1.0` exists, code wants `>=1.0.0`) — that's why we use the
transformers route instead.

### YOLO

`ultralytics 8.4.24` with **YOLO26** weights (`yolo26n.pt` …
`yolo26x.pt`, plus `-seg` variants). Weights auto-download from
`ultralytics/assets` on first use into the project root.

---

## UI tour

### Main page (`ui/app.py`)

Sidebar groups, top to bottom:

- **Camera mode**: static / moving / auto — feeds `FrameSelector`'s
  motion-compensation strategy
- **Object Detection (YOLO26)**: enable, sweep FPS, model size
  (n/s/m/l/x), seg variant toggle, confidence
- **Depth Estimation**: enable
- **Audio (whisper.cpp)**: enable, model dropdown, GPU radio
  (NVIDIA / AMD / CPU)
- **VLM Captioning (llama-server)**: frame source (keyframes / scored),
  caption FPS, frames per chunk, "Keep VLM running after this video"
- **Frame Selection**: keyframe threshold (× σ)

Main area:

- File uploader (configured to allow up to 4 GB via
  `.streamlit/config.toml`)
- **Preview frame selection** button → renders the motion-delta plot with
  keyframes marked
- **Run full pipeline** button → executes the v2 pipeline. Above the live
  stage log you get a JS-driven **elapsed timer + ETA + progress bar**
  with a collapsible per-stage cost breakdown. After completion you get
  the actual time and how it compared to the estimate.

### Results page (`ui/pages/Results.py`)

Sidebar dropdown: pick any past run from `results/`.

8 tabs:

1. **Overview** — metrics, summary, entities table, plotly Gantt timeline
   (x-axis in real video MM:SS), thumbnail grid with bbox overlay
2. **Frame inspector** — slider scrubs through every frame; per-frame
   detection table; events active at that timestamp
3. **Entity inspector** — pick E1/E2/…; events involving that entity;
   filtered timeline; lifespan-window thumbnails
4. **VLM (Qwen) outputs** — one expander per chunk, with the parsed
   `FRAME_<idx>:` lines, the propagated `SUMMARY:` seed, and the full
   raw Qwen response
5. **Captions** — full transcript, segment table, download `transcript.json`
6. **Interactive timeline** — embedded HTML widget with annotated video +
   Plotly chart side by side; hover an event to see its description, click
   to seek the video to that segment with optional loop
7. **Annotated video** — build / rebuild an MP4 from saved frames with
   bbox + depth overlays; plays via `st.video`
8. **Files** — full file listing of the run dir, embedded pyvis graph,
   download `analysis.json`

---

## CLI

Equivalent of clicking everything in the UI:

```bash
/home/ashie/anaconda3/envs/msd/bin/python -m src.pipeline_v2 \
  --video clip.mp4 \
  --output results/clip.mp4 \
  --camera auto \
  --yolo-size n --seg --yolo-fps 1.0 --confidence 0.5 \
  --depth \
  --audio --whisper-model base.en --whisper-gpu-device 1 \
  --vlm-fps 0.5 --chunk-size 10 \
  --vlm-frame-mode keyframes \
  --vlm-url http://127.0.0.1:8082 \
  --vlm-model Qwen3.5-9B-Q8_0.gguf \
  --vlm-autostart-service qwen35-turboquant.service \
  --vlm-autostop \
  --threshold-mult 1.0
```

Skip-VLM mode (everything but the LLM):

```bash
... --skip-vlm
```

---

## Cost model / ETA estimator

`src/runtime_estimator.py` exposes `estimate_run(...)` which returns
per-stage and total wall time before the run starts. The UI uses this for
the live ETA bar; you can also call it from a notebook:

```python
from src.runtime_estimator import estimate_run, format_seconds
est = estimate_run("clip.mp4", yolo_fps=1, vlm_fps=0.5, chunk_size=10,
                   yolo_size="n", audio_enabled=True, whisper_model="base.en")
for s in est.stages:
    print(f"{s.name:24s} {format_seconds(s.seconds):>6s}  {s.note}")
print(f"{'TOTAL':24s} {format_seconds(est.total_s):>6s}")
```

The numbers are calibrated for the 3060 + 9070 XT setup; adjust constants
at the top of `runtime_estimator.py` if running elsewhere.

---

## Tests

```bash
pytest tests/ --ignore=tests/test_vram_budget.py
```

29 tests as of this writing. The vram-budget test is skipped because it
references a `torch._C._CudaDeviceProperties.total_mem` attribute that
doesn't exist in modern torch (renamed to `total_memory`).

---

## Known issues / blockers

- **`depth-anything-v2>=1.0.0` in `requirements.txt`** is unsatisfiable on
  PyPI (only `0.1.0` exists). The pipeline doesn't actually need it
  anymore (uses `transformers` directly), but `pip install -r
  requirements.txt` still fails on that line. Loosen to `>=0.1.0` or
  remove the dep.
- **AMD radv `vk::DeviceLostError`** intermittently when whisper-cli runs
  on Vulkan device 0. Default is device 1 (NVIDIA) for that reason. If
  you must use AMD for whisper, expect occasional crashes.
- **Streamlit upload limit raised to 4 GB** via `.streamlit/config.toml`.
  Python briefly holds the entire upload in RAM during `uploaded.read()`.
  Multi-user concurrent uploads on a memory-constrained box will OOM.
- **Filename-keyed output dirs** mean uploading two different videos with
  the same filename overwrites the first run's results.
- **Mask persistence**: the seg variant of YOLO computes instance masks
  but they're discarded — only bboxes go into the DB. Toggle exists for
  future work; until masks are persisted, seg mode just costs ~15 % more
  inference for no extra data.

---

## Repository layout

```text
TempoGraph/
├── Makefile                          # install / run / test targets
├── bootstrap.sh                      # one-shot installer
├── QUICKSTART.md                     # 1-line install guide
├── README.md
├── docs/PIPELINE.md                  # deep technical doc
├── AGENTS.md                         # coding standards for agents
├── requirements.txt
├── .streamlit/config.toml            # 4 GB upload cap, telemetry off
├── src/
│   ├── pipeline_v2.py                # orchestrator
│   ├── aggregator.py                 # chunk → analysis.json (also takes transcript)
│   ├── runtime_estimator.py          # ETA model used by the UI
│   ├── batch_runner.py               # batch every video in a directory
│   ├── dataset_exporter.py           # COCO + JSONL dataset export
│   ├── graph_builder.py
│   ├── json_parser.py
│   ├── models.py
│   ├── storage.py                    # SQLite schema + helpers
│   ├── backends/
│   │   ├── base.py
│   │   └── llama_server_backend.py   # → llama.cpp /v1/chat/completions
│   └── modules/
│       ├── depth.py                  # transformers depth-anything-v2
│       ├── detector.py               # ultralytics YOLO26
│       ├── frame_scorer.py           # top-K scorer
│       ├── frame_selector.py         # motion-aware selector
│       └── whisper_cpp.py            # whisper.cpp subprocess wrapper
├── ui/
│   ├── app.py                        # main pipeline page (Streamlit)
│   └── pages/
│       └── Results.py                # results browser (Streamlit)
├── scripts/
│   └── smoke_test_v2.py              # end-to-end pipeline smoke test
├── tools/
│   └── make_test_video.py            # synthetic-video generator
└── tests/
```

## License

MIT
