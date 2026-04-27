# TempoGraph v2 — Detailed Pipeline Documentation

This is the deep-dive companion to `README.md`. Where the README tells you
how to run the thing, this doc tells you what every stage does, what
knobs it exposes, what it persists, and which `file:line` to read for
the actual code.

Last verified against the codebase: 2026-04-27.

---

## Table of contents

1. [Architecture & hardware split](#architecture--hardware-split)
2. [Stage-by-stage walkthrough](#stage-by-stage-walkthrough)
   - [Stage 1 — Frame selection](#stage-1--frame-selection)
   - [Stage 1.5 — Audio transcription](#stage-15--audio-transcription)
   - [Stage 2 — YOLO detection](#stage-2--yolo-detection)
   - [Stage 3 — Depth estimation](#stage-3--depth-estimation)
   - [Stage 4 — Frame scoring](#stage-4--frame-scoring-vlm-frame-pick)
   - [Stage 5 — VLM captioning (chunked)](#stage-5--vlm-captioning-chunked)
   - [Stage 6 — Aggregation](#stage-6--aggregation)
   - [Stage extras — VLM autostart / autostop](#stage-extras--vlm-autostartautostop)
3. [The qwen llama-server integration](#the-qwen-llama-server-integration)
4. [The whisper.cpp integration](#the-whispercpp-integration)
5. [Storage schema](#storage-schema)
6. [Output files per run](#output-files-per-run)
7. [The aggregator's prompt](#the-aggregators-prompt)
8. [Runtime cost model](#runtime-cost-model)
9. [UI walkthrough](#ui-walkthrough)
10. [CLI reference](#cli-reference)
11. [Known issues / footguns](#known-issues--footguns)
12. [Glossary](#glossary)

---

## Architecture & hardware split

TempoGraph v2 is a **chunked, mostly-local, multimodal video-analysis
pipeline**. Designed for a single workstation with two discrete GPUs:

| GPU | API | Role | Stages it runs |
|---|---|---|---|
| NVIDIA RTX 3060 Mobile (6 GiB) | CUDA + Vulkan | Vision + ASR | Stage 1.5 (Whisper Vulkan), Stage 2 (YOLO CUDA), Stage 3 (Depth CUDA) |
| AMD RX 9070 XT (16 GiB) | ROCm/HIP | LLM serving | Stage 5 (Qwen3.5-VL via llama-server), Stage 6 (Qwen3.5 text via llama-server) |

Splitting them prevents memory contention. The torch in the `msd` conda
env is CUDA-built so it sees only the NVIDIA card; the AMD card is owned
exclusively by `llama-cpp-turboquant/build/bin/llama-server` (HIP build).
Whisper.cpp is built with `-DGGML_VULKAN=1` and uses Vulkan device 1 (=
NVIDIA) by default because radv on gfx1201 occasionally throws
`vk::DeviceLostError`.

The orchestrator lives in `src/pipeline_v2.py:PipelineV2.run()`; it walks
the stages strictly in order, persists every intermediate result to the
per-run SQLite at `<run_dir>/tempograph.db`, and emits per-stage events
through an optional `on_stage(name, event, info)` callback that the
Streamlit UI consumes for the live progress display.

```
┌──────────────────┐    ┌──────────────────┐    ┌───────────────────┐
│  source video    │───▶│  PipelineV2.run  │───▶│  results/<name>/  │
│  (.mp4/.avi/...)│    │  (orchestrator)  │    │  ├─ tempograph.db │
└──────────────────┘    └──────────────────┘    │  ├─ analysis.json │
                                  │              │  ├─ chunks.json   │
        ┌─────────────────────────┴────────┐     │  ├─ transcript.. │
        ▼                                  ▼     │  ├─ frames/      │
┌──────────────┐    Stages 1.5/2/3 use    │     │  ├─ depth/       │
│ NVIDIA 3060  │   ◀── Whisper, YOLO,     │     │  └─ ...          │
│ (CUDA + Vk)  │       Depth Anything V2  │     └───────────────────┘
└──────────────┘                          │
                                          ▼
┌──────────────┐    Stage 5/6 hit
│  AMD 9070 XT │   ◀── llama-server :8082
│  (ROCm/HIP)  │       (Qwen3.5-VL)
└──────────────┘
```

---

## Stage-by-stage walkthrough

Stage numbers match the `_stage(...)` callback names in
`src/pipeline_v2.py`.

### Stage 1 — Frame selection

**Code**: `src/modules/frame_selector.py:FrameSelector.select()`,
called from `src/pipeline_v2.py:170-188`.

**What it does**: scans the video at `yolo_fps` (the **Sweep FPS**
slider, default 1 fps), computes a per-frame motion delta, and returns:

- `scan_indices` — every frame index that was sampled
- `deltas` — the motion delta for each scanned frame
- `frame_indices` — the deduplicated subset that will actually go on to
  YOLO. Same as `scan_indices` for static-camera mode; for
  moving-camera mode it does homography-compensated motion estimation
  and merges very-close frames.
- `keyframe_indices` — the subset whose delta exceeded
  `mean(deltas) + threshold_mult × std(deltas)`. These are the "motion
  peaks" — visible as **green** dots in the Frame Selection preview.
- `sampled_indices` — non-keyframe sampled frames (orange dots in the
  preview).

**Camera modes** (sidebar **Camera mode**):
- `static`: pure pixel-diff between consecutive sampled frames
- `moving`: estimates homography (camera motion), warps prior frame
  before subtracting, so only *content* motion contributes to the delta
- `auto`: heuristic picks one based on the first few frames

**Output**: After the FrameSelector returns, `pipeline_v2.py:190-197`:
- saves a JPEG per selected frame to `<run_dir>/frames/frame_<NNNNNN>.jpg`
  (downscaled to `frame_max_width=640` if larger; aspect ratio preserved)
- inserts one row per frame into `frames` table with
  `(frame_idx, timestamp_ms, image_path, is_keyframe, delta_score)`

**Stage event emitted**:
```
✓ Frame selection — done  (elapsed_s=0.3, frames=60, keyframes=12)
```

---

### Stage 1.5 — Audio transcription

**Code**: `src/modules/whisper_cpp.py:WhisperCppTranscriber`,
`src/pipeline_v2.py:200-228`. Optional — gated on `audio_enabled`.

**What it does**:
1. Extracts the **entire audio track** from the source video using ffmpeg:
   ```
   ffmpeg -y -i <video> -vn -ar 16000 -ac 1 -c:a pcm_s16le /tmp/audio.wav
   ```
   — 16 kHz mono PCM, no time limit. Operates on the full video, not
   the sampled frames.
2. Calls the whisper.cpp binary:
   ```
   /home/ashie/whisper.cpp/build/bin/whisper-cli \
     -m /home/ashie/whisper.cpp/models/ggml-<model>.bin \
     -f /tmp/audio.wav \
     -oj -of /tmp/audio \
     -dev <gpu_device>      # 0=AMD, 1=NVIDIA, default 1
                            # OR --no-gpu for CPU
   ```
3. Parses the resulting `audio.json` (whisper.cpp's `transcription[]`
   array with `offsets.{from,to}` in ms) into a list of
   `WhisperSegment` dataclasses.
4. Persists each segment to the `audio_segments` table:
   ```sql
   audio_segments(segment_id PK, start_ms, end_ms, text,
                  no_speech_prob, avg_logprob)
   ```
5. Writes a sidecar `<run_dir>/transcript.json` for fast UI loading.

If the whisper binary or model is missing, `ensure_model_downloaded()`
runs `bash models/download-ggml-model.sh <name>` to fetch the ~75 MB to
~3 GB model from Hugging Face.

**Available models** (sidebar **Whisper model**):

| name | size | rough realtime ratio on 3060 (Vulkan) |
|---|---:|---|
| `tiny` / `tiny.en` | ~75 MB | ~32× rt |
| `base` / `base.en` | ~141 MB | ~16× rt **(default)** |
| `small` / `small.en` | ~466 MB | ~6× rt |
| `medium` / `medium.en` | ~1.5 GB | ~2× rt |
| `large-v1` / `-v2` / `-v3` | ~3 GB | ~1× rt |
| `large-v3-turbo` | ~1.6 GB | ~4× rt |

`.en` variants are English-only and slightly faster/more accurate on
English speech. Multilingual variants auto-detect the language unless
you pass `--whisper-language en`.

**Stage event**:
```
✓ Audio transcription — done  (elapsed_s=2.3, segments=14, chars=1872)
```

---

### Stage 2 — YOLO detection

**Code**: `src/modules/detector.py:ObjectDetector.detect_to_db()`,
`src/pipeline_v2.py:230-260`.

**What it does**: runs ultralytics YOLO over the saved JPEGs from
Stage 1, persists every detection to the `detections` table.

**Models** (sidebar **Model size** + **Use segmentation variant**):

| size | bbox-only | seg variant | weights | typical 3060 latency |
|---|---|---|---|---|
| `n` | `yolo26n.pt` | `yolo26n-seg.pt` | ~5 MB | ~25 ms/frame |
| `s` | `yolo26s.pt` | `yolo26s-seg.pt` | ~22 MB | ~45 ms/frame |
| `m` | `yolo26m.pt` | `yolo26m-seg.pt` | ~50 MB | ~80 ms/frame |
| `l` | `yolo26l.pt` | `yolo26l-seg.pt` | ~85 MB | ~130 ms/frame |
| `x` | `yolo26x.pt` | `yolo26x-seg.pt` | ~140 MB | ~220 ms/frame |

Vocabulary: closed COCO-80. Weights auto-download from
`ultralytics/assets` GitHub release on first use into the project root.

**Bbox storage**: each detection is saved with **normalised** coordinates
(0..1) in `(x1, y1, x2, y2)`, normalised against the **saved JPEG
dimensions** — not the source video dimensions. So
`x_pixel = x_norm × jpeg_width` always works in the UI. This was a bug
that's been fixed; runs older than 2026-04-27 may have wrong bboxes
because they were normalised against source dims.

**Seg vs bbox**: only one model loads per run. The seg variant computes
masks alongside bboxes in the same forward pass, but **masks are
discarded** — only bboxes go into the DB. The toggle is in place for
future mask-persistence work.

**Schema**:
```sql
detections(
  detection_id PK AUTOINCREMENT,
  frame_idx FK,
  track_id INTEGER,            -- when tracking is on; null otherwise
  class_name TEXT,
  x1, y1, x2, y2 REAL,         -- normalised to JPEG dims, [0,1]
  confidence REAL,
  mean_depth REAL              -- populated by Stage 3 if depth runs
)
```

**Stage event**:
```
✓ YOLO detection — done  (elapsed_s=1.4, detections=134)
```

---

### Stage 3 — Depth estimation

**Code**: `src/modules/depth.py:DepthEstimator.estimate_to_db()`,
`src/pipeline_v2.py:262-285`. Optional — gated on `depth_enabled`.

**What it does**: runs Depth Anything V2 (small, vits variant) on each
saved JPEG, writes the depth map to `<run_dir>/depth/depth_<NNNNNN>.npy`
(normalised [0,1] float32), inserts a row in `depth_frames`, and updates
each `detections` row in the same frame with `mean_depth = mean(depth
within the bbox)`.

**Loaded via**: the `transformers` pipeline:
```python
pipe = pipeline("depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0)  # cuda:0 = 3060
```

A small adapter (`_DepthAnythingPipelineAdapter` in `depth.py`) makes the
HF pipeline look like the upstream `DepthAnythingV2` class so the rest
of the depth code didn't have to change.

The original `depth-anything-v2` PyPI package is unsatisfiable
(`requirements.txt` pins `>=1.0.0`, only `0.1.0` exists), which is why
we go through `transformers` instead.

**Available variants** in `_HF_REPOS` (currently only vits is wired
through the UI; vitb/vitl require a small code change):

| variant | params | VRAM | speed |
|---|---:|---|---|
| `vits` (default) | 25M | ~0.5 GB | ~50 ms/frame |
| `vitb` | 97M | ~1.5 GB | ~100 ms/frame |
| `vitl` | 335M | ~4 GB | ~250 ms/frame |

**Stage event**:
```
✓ Depth estimation — done  (elapsed_s=4.9)
```

---

### Stage 4 — Frame scoring (VLM frame pick)

**Code**: `src/modules/frame_scorer.py:FrameScorer`,
`src/pipeline_v2.py:295-330`.

**What it does**: picks **which frames** the VLM will see. Two modes
controlled by the sidebar **Frame source for VLM** radio:

#### Mode `keyframes` (default after my recent change)
```python
vlm_frames = sorted(set(selection.keyframe_indices))
```
Only frames that FrameSelector flagged as keyframes (motion peaks). The
`vlm_fps` slider is ignored in this mode. If FrameSelector returned
zero keyframes, falls back to all sampled frames so the VLM has
something to look at.

#### Mode `scored`
```python
k = round(video_duration_s × vlm_fps)
vlm_frames = FrameScorer().score_and_select(
    candidate_frame_indices=selection.frame_indices,
    keyframe_indices=set(selection.keyframe_indices),
    k=k,
)
```

`FrameScorer` ranks every YOLO-scanned frame using a weighted score:

```
score = α·delta + β·class_set_change + γ·track_churn + ε·iou_drop
        with α=1.0, β=2.0, γ=2.0, ε=0.5
```

| signal | what it measures |
|---|---|
| `delta` | motion delta from FrameSelector |
| `class_set_change` | new YOLO class names appearing or disappearing vs prev frame |
| `track_churn` | tracked IDs entering or leaving |
| `iou_drop` | how much existing tracked objects moved (1 - mean IoU) |

Keyframes are pinned in first; remaining slots filled by top-scoring
non-keyframes. Result is sorted in time order.

**Stage event**:
```
✓ Frame scoring — done  (elapsed_s=0.05, mode=keyframes, vlm_frames=12, fallback_used=False)
```

---

### Stage 5 — VLM captioning (chunked)

**Code**: `src/backends/llama_server_backend.py:LlamaServerBackend.caption_chunks()`,
`src/pipeline_v2.py:333-380`.

**What "chunked" means**: the chosen `vlm_frames` list (from Stage 4) is
sliced into consecutive groups of `chunk_size` (sidebar **Frames per
chunk**, default 10). Each chunk = one HTTP POST to
`http://127.0.0.1:8082/v1/chat/completions`.

```python
chunks = []
for i in range(0, len(vlm_frames), chunk_size):
    chunks.append((len(chunks), vlm_frames[i : i + chunk_size]))
```

So 30 VLM frames + chunk_size 10 → 3 chunks. **No interleaving, no
balancing** — frames stay in time order so the model can describe
sequences naturally.

#### What goes inside one chunk request

For every frame in the chunk, two things are gathered:

1. The **JPEG image** (base64-encoded, sent as an `image_url` content item)
2. A **text line** with the frame's metadata + YOLO detections:
   ```
   [frame 30 — t=00:01.00] YOLO: person at [0.20,0.30,0.55,0.95] conf=0.81
   ```

Those text lines are spliced into the prompt template
(`CHUNK_PROMPT_TEMPLATE`):

```
You are watching a short segment of a video. Describe what is happening
across these frames in chronological order.

Previous segment summary: {seed}    ← context bridge from prior chunk

For each frame below, output ONE LINE describing the action.
If consecutive frames show no significant change, write "(no change)".
End with ONE LINE summarizing this segment in <= 20 words for use as
context in the next segment.

Frame data:
[frame 0 — t=00:00.00] YOLO: ...
[frame 30 — t=00:01.00] YOLO: ...
... (one line per frame in this chunk)

Output format:
FRAME_<idx>: <description>
...
SUMMARY: <one-line segment summary>
```

The full payload sent to the server:
```json
{
  "model": "Qwen3.5-9B-Q8_0.gguf",
  "max_tokens": 4096,
  "temperature": 0.1,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "<the prompt above>"},
      {"type": "image_url",
       "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
      {"type": "image_url", ...}, ...
    ]
  }]
}
```

`enable_thinking: false` is **critical** — Qwen3 has a hidden reasoning
channel that otherwise eats the entire `max_tokens` budget on internal
reasoning and returns `content: ""`. The `/no_think` prefix in the
prompt is a belt-and-suspenders.

#### What comes back

```
FRAME_0: a person stands on the left side of the frame
FRAME_30: the person takes a step toward the right
FRAME_60: the person picks up a ball
SUMMARY: a person walks right and picks up a ball
```

Parsed in `_parse_chunk_response()` with two regexes:
- `^FRAME[_ ]?(\d+)\s*[:\-]\s*(.+)$` → per-frame lines
- `^SUMMARY\s*[:\-]\s*(.+)$` → segment summary

#### Seed propagation

```python
if summary:
    seed = summary    # next chunk's prompt gets this
```

The summary line becomes the `Previous segment summary:` for chunk N+1.
That's how the stateless API carries narrative context across chunks.

#### Persistence

After all chunks complete, `pipeline_v2.py:354-372` writes
`<run_dir>/chunks.json`:

```json
[
  {
    "chunk_id": 0,
    "frame_indices": [12, 30, 60, 90, 120, 150],
    "per_frame_lines": {"12": "...", "30": "...", ...},
    "summary": "stick figure walks right and picks up a ball",
    "raw_response": "FRAME_12: ...\n... \nSUMMARY: ..."
  },
  ...
]
```

#### Per-chunk telemetry

The backend exposes an `on_chunk` callback that fires after every
chunk's HTTP response. The pipeline plumbs it into the stage callback as
a `"VLM chunk"` event:

```
{
  "chunk_id": 0, "chunk_index": 0, "n_total": 3,
  "n_images": 10,
  "prompt_tokens": 3247,
  "completion_tokens": 124,
  "total_tokens": 3371,
  "n_ctx": 100096,                  # from /props
  "elapsed_s": 4.4,
  "ok": true
}
```

The UI consumes these to render the live **VLM context window usage**
panel (per-chunk progress bars colored blue/orange/red based on % of
n_ctx).

#### Service auto-start

If `is_available()` returns false at the start of Stage 5, the pipeline
calls `_ensure_vlm_ready()` which runs `systemctl --user start
qwen35-turboquant.service` and polls `/v1/models` for up to
`vlm_autostart_timeout_s` (default 90 s). On success, fires `VLM
autostart — done`. On timeout or systemctl failure, raises a clean
RuntimeError with the systemctl command to run.

The matching `_maybe_stop_vlm_service()` runs in the `finally` block at
the end of `run()` if `vlm_autostop=True`. UI sets this from the
sidebar checkbox **Keep VLM running after this video**.

---

### Stage 6 — Aggregation

**Code**: `src/aggregator.py:CaptionAggregator.aggregate()`,
`src/pipeline_v2.py:388-410`.

**What it does**: takes all the per-chunk outputs from Stage 5, plus the
audio transcript (if Stage 1.5 ran), and makes **one final text-only
call** to the same llama-server (now without images) asking it to
synthesise everything into the structured `AnalysisResult` JSON schema.

Two paths based on chunk count:

- `len(chunks) <= single_pass_max_chunks` (default 30) — flatten all
  chunk lines and summaries into one log, send in one call.
- otherwise — **hierarchical compression** in `_compress_hierarchical()`:
  partition chunks into groups of `group_size=10`, compress each group's
  summaries to a paragraph via `META_PROMPT`, concatenate the paragraphs,
  then send as the input to the final synthesis call. Keeps very long
  videos within the 100k context window.

Final synthesis prompt (`SINGLE_PASS_PROMPT`):

```
You are given a chronological log of per-frame and per-chunk
descriptions of a video, and (optionally) a speech transcript. Identify
entities (people, animals, vehicles, objects), their behaviors and
interactions over time, and produce structured JSON. If a transcript is
provided, also populate audio_events and multimodal_correlations linking
what is said to what is seen.

Schema:
{"entities":[...],
 "visual_events":[...],
 "audio_events":[{"type":"speech","start_time":"MM:SS","end_time":"MM:SS","text":"...","speaker":"unknown"}],
 "multimodal_correlations":[{"audio_idx":0,"visual_idx":2,"description":"speaker says X while subject does Y","confidence":0.7}],
 "summary":"..."}

Caption log:
{captions}

Audio transcript (may be empty):
{transcript}

Output ONLY the JSON.
```

The response is parsed by `JSONParser` (lenient — strips `<think>`
blocks, extracts embedded JSON, fixes common issues), and the result is
serialised to `<run_dir>/analysis.json`.

A `pyvis` graph is built from the entities and saved to
`<run_dir>/graph.html` if pyvis is installed.

The aggregator's call also fires an `on_call` callback with
`prompt_tokens / completion_tokens / total_tokens` — the UI shows this
as the "aggregator" row in the context-window panel. This is the call
most likely to push toward the 100k ceiling on long videos.

---

### Stage extras — VLM autostart/autostop

These are not numbered stages but they fire as their own stage events:

- **VLM autostart** — fires only if the qwen service was down at the
  start of Stage 5. Shows time spent waiting for the model to load (~10
  – 15 s typical for Qwen3.5-9B Q8_0 on the AMD card).
- **VLM autostop** — fires after the run if `vlm_autostop=True`. Frees
  the AMD VRAM so other apps can use the card.

Both are gated by `vlm_autostart_service` being non-None — which the
UI sets to `"qwen35-turboquant.service"` automatically. CLI users opt in
via `--vlm-autostart-service qwen35-turboquant.service --vlm-autostop`.

---

## The qwen llama-server integration

### Service definition

```ini
# ~/.config/systemd/user/qwen35-turboquant.service
[Unit]
Description=Qwen3.5 9B (Vision) TurboQuant LLM Server
After=network.target

[Service]
Type=simple
ExecStart=/home/ashie/llama-cpp-turboquant/build/bin/llama-server \
  -m /home/ashie/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf \
  --mmproj /home/ashie/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/mmproj-Qwen3.5-9B-BF16.gguf \
  -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -c 100000 \
  -np 1 \
  --host 0.0.0.0 --port 8082
Restart=on-failure
RestartSec=10
Environment=HSA_OVERRIDE_GFX_VERSION=12.0.1
Environment=HIP_VISIBLE_DEVICES=0
WorkingDirectory=/home/ashie/llama-cpp-turboquant

[Install]
WantedBy=default.target
```

Key bits:
- The binary `llama-cpp-turboquant/build/bin/llama-server` is the **HIP
  build** (has `libggml-hip.so` next to it) → runs on AMD GPU, not CUDA.
- `--mmproj` is what enables vision. Without it, the same model loads
  but `/props` reports `modalities: {vision: false}` and image input is
  rejected.
- `-c 100000` is the **1 lakh** context window. `--cache-type-k turbo3
  --cache-type-v turbo3` is a custom 3-bit KV-cache quant from the
  turboquant fork that keeps the cache footprint manageable at 100k.

### API used by the backend

| endpoint | who calls it | what we extract |
|---|---|---|
| `GET /v1/models` | `is_available()` | health check |
| `GET /props` | `get_n_ctx()` | `default_generation_settings.n_ctx` |
| `POST /v1/chat/completions` | `caption_chunks()`, `aggregator._call_llm_text()` | `choices[0].message.content`, `usage.{prompt,completion,total}_tokens` |

Standard OpenAI-style payload + `chat_template_kwargs:{enable_thinking:
false}` to disable Qwen3's hidden reasoning channel.

### Context-window math (important)

Per chunk request, prompt token cost ≈

```
~200 (template + seed)
 + ~50 × n_frames_in_chunk (per-frame YOLO text lines)
 + ~300 × n_frames_in_chunk (image tokens — Qwen3.5-VL)
 ≈ 200 + 350 × chunk_size
```

So `chunk_size=10` → ~3.7 k tokens. `chunk_size=30` → ~10.7 k. Plenty of
headroom in a 100k window.

The aggregator's final call is text-only and grows linearly with chunk
count — that's the one to watch on long videos. Hierarchical
compression (Stage 6 path B) kicks in automatically beyond 30 chunks.

---

## The whisper.cpp integration

### Build

```bash
cd /home/ashie/whisper.cpp
cmake -B build -DGGML_VULKAN=1
cmake --build build -j --config Release
bash models/download-ggml-model.sh base.en   # one-time per model
```

Vulkan was chosen over CUDA because the system doesn't have the CUDA
toolkit (`nvcc`) installed — Vulkan needs no toolchain beyond the loader
already present from the llama-cpp-turboquant build.

### Devices

`vulkaninfo --summary` on this box shows:
```
device 0: AMD Radeon RX 9070 XT (RADV GFX1201)
device 1: NVIDIA GeForce RTX 3060 Laptop GPU
device 2: Intel Iris Xe Graphics (iGPU)
device 3: llvmpipe (CPU fallback)
```

Default is **device 1 (NVIDIA)** because device 0 (AMD radv) hits
intermittent `vk::DeviceLostError` on gfx1201 during whisper inference.
Sidebar lets you switch to AMD or to CPU (`--no-gpu`) if you want.

### Subprocess invocation

```
whisper-cli -m models/ggml-<model>.bin \
  -f /tmp/audio.wav \
  -oj -of /tmp/audio \
  -dev <gpu>          # OR --no-gpu
  [-l <language>]
  [-t <threads>]
```

Output JSON shape:
```json
{
  "transcription": [
    {"offsets": {"from": 0, "to": 11000},
     "text": "And so my fellow Americans..."},
    ...
  ]
}
```

`from`/`to` are in milliseconds. Each row becomes a row in
`audio_segments`.

---

## Storage schema

Single SQLite at `<run_dir>/tempograph.db`. Full schema from
`src/storage.py`:

```sql
CREATE TABLE IF NOT EXISTS frames (
    frame_idx INTEGER PRIMARY KEY,
    timestamp_ms INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    is_keyframe INTEGER NOT NULL,
    delta_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_idx INTEGER NOT NULL,
    track_id INTEGER,
    class_name TEXT NOT NULL,
    x1 REAL NOT NULL,           -- normalised to JPEG dims, [0,1]
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    confidence REAL NOT NULL,
    mean_depth REAL,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);

CREATE TABLE IF NOT EXISTS depth_frames (
    frame_idx INTEGER PRIMARY KEY,
    depth_npy_path TEXT NOT NULL,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);

CREATE TABLE IF NOT EXISTS audio_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    no_speech_prob REAL,
    avg_logprob REAL
);

CREATE INDEX IF NOT EXISTS idx_det_frame   ON detections(frame_idx);
CREATE INDEX IF NOT EXISTS idx_audio_start ON audio_segments(start_ms);
```

Useful queries:

```sql
-- "How many detections per frame?"
SELECT frame_idx, COUNT(*) FROM detections GROUP BY frame_idx;

-- "What was being said when frame N was captured?" (no schema change needed)
SELECT a.text FROM audio_segments a JOIN frames f
  ON a.start_ms <= f.timestamp_ms AND a.end_ms > f.timestamp_ms
  WHERE f.frame_idx = ?;

-- "All keyframes with their detection summaries"
SELECT f.frame_idx, f.timestamp_ms,
       GROUP_CONCAT(d.class_name, ',') AS classes
  FROM frames f LEFT JOIN detections d USING (frame_idx)
  WHERE f.is_keyframe = 1
  GROUP BY f.frame_idx
  ORDER BY f.timestamp_ms;
```

---

## Output files per run

```
results/<video_filename>/
├── tempograph.db        SQLite, all 4 tables above
├── frames/
│   ├── frame_000000.jpg ← downscaled to frame_max_width=640
│   ├── frame_000030.jpg
│   └── ...
├── depth/                ← only if --depth
│   ├── depth_000000.npy ← float32, normalised [0,1]
│   └── ...
├── transcript.json       ← only if --audio (sidecar of audio_segments)
├── chunks.json           ← per-chunk Qwen output (raw + parsed)
├── analysis.json         ← final structured analysis (AnalysisResult)
├── annotated_strip.mp4   ← built on demand from Results UI
└── graph.html            ← pyvis graph if pyvis is installed
```

---

## The aggregator's prompt

Two-mode prompt selection in `aggregator.py`:

| chunks count | path | implementation |
|---|---|---|
| ≤ `single_pass_max_chunks` (30) | **single-pass** | `_single_pass()` — concatenate per-frame and per-chunk lines, send in one call |
| > 30 | **hierarchical** | `_compress_hierarchical()` — group into chunks of `group_size=10`, summarise each group, then single-pass on the summaries |

Both paths inject the audio transcript as a separate block in the
prompt, formatted as:

```
[00:00.00-00:11.00] And so my fellow Americans, ask not...
[00:11.50-00:18.20] ask not what your country can do for you...
```

The schema in the prompt explicitly includes `audio_events` and
`multimodal_correlations` so the LLM is invited to link audio events to
visual events.

---

## Runtime cost model

`src/runtime_estimator.py:estimate_run()` returns a per-stage cost
breakdown and a total. Used by the UI to show the live ETA. Constants
calibrated for the 3060 + 9070 XT setup; tunable at the top of the file
if you run it elsewhere.

```python
from src.runtime_estimator import estimate_run, format_seconds
est = estimate_run(
    video_path="clip.mp4",
    yolo_fps=1.0, vlm_fps=0.5, chunk_size=10,
    yolo_size="n", use_segmentation=True,
    depth_enabled=False,
    audio_enabled=True, whisper_model="base.en",
    vlm_frame_mode="keyframes",
    vlm_autostart_cold=True,
)
print(format_seconds(est.total_s))
for s in est.stages:
    print(f"  {s.name:24s} {format_seconds(s.seconds):>6s}  ({s.note})")
```

Sample output for a 10 s clip:

```
0:30
  Frame selection            0:00  (10 sampled frames)
  Audio transcription        0:01  (base.en on 10.0s of audio)
  YOLO detection             0:01  (yolo26n-seg × 10)
  Frame scoring              0:00  (5 VLM frames)
  VLM autostart              0:12  (qwen3.5-9B model load on AMD)
  VLM captioning             0:11  (1 chunks × ~11.0s)
  Aggregation                0:04  (single text-only LLM call)
```

---

## UI walkthrough

### Main page (`ui/app.py`)

Sidebar groups, top to bottom:

| group | controls |
|---|---|
| Camera | static / moving / auto |
| Object Detection (YOLO26) | enable, sweep FPS, model size (n…x), seg toggle, confidence |
| Depth Estimation | enable |
| Audio (whisper.cpp) | enable, model dropdown (12 variants), GPU radio |
| VLM Captioning (llama-server) | frame source (keyframes/scored), caption FPS, frames per chunk, "Keep VLM running" |
| Frame Selection | keyframe threshold (× σ) |

Main area:
- File uploader (4 GB cap via `.streamlit/config.toml`)
- **Preview frame selection** button → motion-delta plot with keyframes
  marked
- **Run full pipeline** button → executes pipeline, shows live progress

Live progress widgets (in order, top to bottom during a run):
1. **Elapsed timer + ETA bar** (HTML/JS panel) — ticks browser-side
   every 200 ms even while Python is blocked. Includes a collapsible
   per-stage cost breakdown.
2. **VLM context window usage** (HTML markdown panel) — appears once
   Stage 5 starts. One row per chunk, colored bars (blue < 60% < orange
   < 85% < red), peak indicator in the top-right. Aggregator gets its
   own row at the bottom.
3. **Stage log** — running list of stage events with icons:
   ▶ start, ✓ done, · skipped, ✗ error.
4. **Final summary** — `Done in 0:34 (ETA was 0:30, +0:04 = 113% of ETA)`

### Results page (`ui/pages/Results.py`)

8 tabs (sidebar nav: pick the run from the dropdown):

| tab | what's there |
|---|---|
| **Overview** | metrics, summary, entities table, plotly Gantt timeline (MM:SS x-axis, full source span), thumbnail grid with bbox + depth overlays |
| **Frame inspector** | scrubber slider through every frame; per-frame detection table; events active at that timestamp |
| **Entity inspector** | pick E1/E2/…; events involving that entity; filtered timeline; thumbnails from the entity's lifespan window |
| **VLM (Qwen) outputs** | one expander per chunk: parsed FRAME lines, propagated SUMMARY seed, full raw Qwen response |
| **Captions** | full transcript paragraph, per-segment table (start/end MM:SS, duration, text, no_speech_prob), download `transcript.json` |
| **Interactive timeline** | embedded HTML widget: annotated video player + Plotly chart side by side; hover bar → summary, click → seek video to that segment, ↻ loop event toggle |
| **Annotated video** | build/rebuild MP4 from saved frames with bbox + depth overlays + frame # / timestamp watermark |
| **Files** | full file listing of run dir, embedded pyvis graph, download `analysis.json` |

---

## CLI reference

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

Skip-VLM mode (everything but the LLM stages):
```bash
... --skip-vlm
```

All flags:

| flag | default | scope |
|---|---|---|
| `--video` | required | input |
| `--output` | `results/v2_run` | output dir |
| `--camera` | `static` | frame selection |
| `--yolo-fps` | 1.0 | frame selection sampling rate |
| `--threshold-mult` | 1.0 | keyframe threshold (× σ) |
| `--yolo-size` | `n` | YOLO model size n/s/m/l/x |
| `--seg` | off | YOLO seg variant |
| `--confidence` | 0.5 | YOLO conf threshold |
| `--depth` | off | run depth stage |
| `--audio` | off | run whisper.cpp stage |
| `--whisper-model` | `base.en` | whisper model |
| `--whisper-binary` | `/home/ashie/whisper.cpp/build/bin/whisper-cli` | path |
| `--whisper-gpu-device` | 1 | Vulkan device id |
| `--whisper-language` | autodetect | force language |
| `--vlm-fps` | 0.5 | VLM caption FPS (scored mode only) |
| `--chunk-size` | 10 | frames per VLM request |
| `--vlm-frame-mode` | `scored` | `scored` or `keyframes` |
| `--vlm-url` | `http://127.0.0.1:8082` | llama-server URL |
| `--vlm-model` | `Qwen3.5-9B-Q8_0.gguf` | model name in payload |
| `--vlm-autostart-service` | None | systemd unit to start if down |
| `--vlm-autostop` | off | stop service after run |
| `--skip-vlm` | off | stop after frame scoring |

---

## Known issues / footguns

1. **`requirements.txt` is broken** on `depth-anything-v2>=1.0.0` —
   PyPI only has `0.1.0`. The pipeline doesn't use that pkg anymore
   (uses `transformers` directly), so just remove or loosen that line if
   you want a clean `pip install -r`.
2. **AMD radv `vk::DeviceLostError`** on Whisper when targeting Vulkan
   device 0. Default is device 1 (NVIDIA) for that reason.
3. **Filename = output dir** — uploading two videos with the same
   filename overwrites the first run. Rename before uploading if you
   care about preserving older runs.
4. **Streamlit upload limit** raised to 4 GB via
   `.streamlit/config.toml`. The whole upload sits in Python RAM during
   `uploaded.read()` — multi-user concurrent uploads on a
   memory-constrained box will OOM.
5. **Mask discard** — the seg variant computes instance masks but they
   aren't persisted (no `mask_*` columns). Toggle exists for future
   work; right now seg mode just costs ~15% more inference for no extra
   stored data.
6. **Pre-2026-04-27 bbox bug** — runs from before that date have bboxes
   normalised against source video dims instead of saved JPEG dims, so
   the UI renders them shrunk by the resize-scale factor. Re-run those
   videos to fix.
7. **Old runs predate audio + chunks.json** — the Captions and VLM
   tabs in the Results page show a friendly "no data for this run"
   message for them.
8. **Aggregator + 100k context** — single-pass aggregator works up to
   ~30 chunks (≈ 60 s × 0.5 fps = 30 frames, 1 chunk per 10 frames =
   3 chunks → easy). Beyond `single_pass_max_chunks=30` it switches to
   hierarchical compression. Watch the aggregator row in the
   context-window panel — if it goes orange/red, hierarchical is
   already saving you.

---

## Glossary

- **Stage event** — one of the dict callbacks fired by `PipelineV2`
  during execution. Shape: `(name: str, event: str, info: dict)` where
  `event ∈ {"start", "done", "skipped", "error"}`. UI consumes these
  for the live log.
- **Keyframe** — a frame where motion delta exceeded `mean + threshold_mult × σ`.
  Always pinned into the VLM frame set in scored mode; defines the
  entire VLM frame set in keyframes mode.
- **Chunk** — a contiguous group of `chunk_size` frames sent to the VLM
  in a single `/v1/chat/completions` request. Chunks 0..N-1 are
  processed sequentially; chunk N's prompt receives chunk N-1's
  `SUMMARY:` line as its `seed`.
- **Seed propagation** — passing the previous chunk's SUMMARY into the
  next chunk's prompt as context. Lets a stateless API maintain
  narrative continuity across chunks.
- **Aggregator** — Stage 6. Single text-only LLM call (no images) that
  takes all per-chunk outputs + the audio transcript and produces the
  final structured `AnalysisResult` (entities, visual_events,
  audio_events, multimodal_correlations, summary).
- **n_ctx** — the LLM server's configured context window size in
  tokens. For our qwen service: 100096 (1 lakh).
- **Hierarchical compression** — aggregator's fallback for very long
  videos. Groups chunks into batches of `group_size=10`, summarises
  each batch, then runs the final synthesis on the summaries instead
  of all per-frame lines. Keeps the aggregator's prompt within n_ctx.
- **Open-vocab** vs **closed-vocab** detector — closed-vocab YOLO
  (yolo26) detects 80 fixed COCO classes; open-vocab YOLO-World accepts
  arbitrary text class names at inference time via a CLIP text encoder.
  Currently only closed-vocab is wired; open-vocab is on the v2
  roadmap for the dataset-generation feature.
