# TempoGraph TODO — post-v3 roadmap

Last updated: 2026-07-11, after the ui-v3-dropflow work (see `SUMMARY.md`).
Supersedes the 2026-04-28 TODO: of that list, dataset-exporter wiring,
per-frame transcript join, stop button, ETA calibration, and resume-from-DB
are ALL DONE (verified, 68 tests). Mask persistence remains and is item 1.

## Instructions for the autonomous agent working this file

Work items top to bottom, one item per session unless told otherwise. For
each item: implement → test → run the item's ACCEPTANCE → commit (imperative
message, one commit per item minimum) → append a dated entry to `SUMMARY.md`.
A component that exists but is never called does not count as done — every
item's acceptance includes an integration-level check. Finish with
`git status --porcelain` empty.

**Environment facts (verified, do not rediscover):**
- Interpreter: `/home/ashie/anaconda3/bin/python3` (streamlit 1.51, pytest,
  torch 2.13+cu130, cv2 4.13, networkx, pyvis). Tests:
  `/home/ashie/anaconda3/bin/python3 -m pytest -q` — currently 68 pass; keep
  them green.
- `ffmpeg`/`ffprobe` on PATH. Sample footage:
  `/home/ashie/Downloads/output_1080p30_brandy.mp4` (copy, never modify).
- Existing run databases for fixtures: `results/*/tempograph.db`.

**FORBIDDEN:** `sudo`, `systemctl`, `pip install`, starting/stopping any
llama-server, pushing to git remotes, allocating memory on the AMD GPU (it
serves the model running you). GPU work only on CUDA device 0 (RTX 3060)
and only where an item explicitly says so. Items marked **[USER-SUPERVISED]**
must NOT be attempted unattended — skip them.

---

## 1. Mask persistence — make the seg models pay their way

**Why:** `yolo26*-seg` variants compute instance masks every frame and throw
them away. Persisting them unlocks mask overlays and future dataset exports.

- Schema (`src/storage.py`): `ALTER TABLE detections ADD COLUMN mask_rle TEXT`
  (guard with a migration that tolerates existing DBs; NULL = no mask).
- Encode: in the detection stage, when the loaded model is a seg variant,
  encode each mask to COCO-style RLE. Write a small pure `src/rle.py`
  (encode/decode, no new deps) with round-trip unit tests.
- Render: Frame inspector (`ui/pages/Results.py`) overlays decoded masks
  (semi-transparent fill, per-entity colour) when `mask_rle` is present;
  annotated-video path gains a `--masks` flag.
- **Acceptance:** RLE round-trip property test (random binary masks); CPU
  smoke run with the seg model on a 5 s synthetic clip → ≥ 1 detection row
  with non-NULL `mask_rle`; AppTest renders the inspector without error on
  a fixture DB containing masks. Suite green.

## 2. Graph-driven clip export — "give me a cut of every event where X"

**Why:** the graph knows what happened when; ffmpeg can cut it. This turns
TempoGraph from an analysis tool into an editing tool.

- `src/clip_export.py`: `select_events(db, entity=None, behavior=None,
  time_range=None) -> [(start_ms, end_ms, label)]` (pad each event ±1.5 s,
  merge overlapping spans) and `export_clips(video_path, spans, out_dir,
  montage=False)` — per-event mp4s via ffmpeg (stream-copy where keyframes
  allow, re-encode fallback); `montage=True` concatenates with crossfade
  and burned-in label lower-thirds.
- UI: "Clips" section in the Graph tab — pick entity and/or behavior from
  existing graph data, preview span list, Export button, download links.
- **Acceptance:** unit tests for span padding/merge math; integration test
  on a fixture DB + synthetic video → N event rows yield N clip files whose
  ffprobe durations match spans ±0.5 s; montage ffprobe-valid; AppTest
  renders the Clips section. Suite green.

## 3. Cross-run entity registry — the archive remembers

**Why:** every run is an island; longitudinal questions ("every appearance
of the brown dog, any video") need identity across runs.

- `src/entity_registry.py` + a global `registry.db` beside `results/`:
  table `canonical_entities(id, label, embedding BLOB, first_seen,
  last_seen, run_count)`. Matching: normalised-label match first; where
  entity crops exist, cosine similarity of torchvision
  `mobilenet_v3_small` features (CPU) with threshold 0.75; else create a
  new canonical id. If weights can't load offline, fall back to label-only
  and say so in SUMMARY.
- Wire: at the end of a pipeline run, upsert entities into the registry
  (config flag `registry_enabled`, default True).
- UI: new "Archive" page — canonical entity list with per-run appearance
  counts; clicking one lists runs + timestamps.
- **Acceptance:** matcher unit tests (same label → same id; distinct → new
  ids; threshold respected with synthetic embeddings); integration: two
  fixture run DBs sharing a label → ONE canonical row, run_count 2; AppTest
  renders Archive page. Suite green.

## 4. Archive-wide Ask (GraphRAG-lite) — depends on item 3

- `src/archive_index.py`: chunk all runs' captions + transcript segments
  (run id + timestamp attached), TF-IDF retriever (sklearn available) with
  the interface designed so an embedding endpoint can replace it later;
  persist the index beside the registry.
- `answer_archive(question)`: top-k chunks across runs → the SAME
  injectable LLM callable the summarizer uses (health-probe resolution;
  heuristic fallback = extractive top-3 chunks with citations).
- UI: "Ask the archive" box on the Archive page; every answer cites
  run + timestamp, clickable through to that run's Results.
- **Acceptance:** retriever unit test (unique planted phrase in fixture run
  A retrieved for a matching query, not from run B); AppTest with fake LLM
  → answer renders with ≥ 1 citation; heuristic path renders without an
  LLM. Suite green.

## 5. Ethogram v2 — from viewer to instrument

- `src/ethology.py`: from event edges → per-entity time budgets (fraction
  of observed time per behavior), behavior transition matrix
  (row-normalised), bout statistics (mean/median bout duration). Pure
  functions over the DB, unit-tested against a hand-built fixture with
  known answers.
- UI: Ethogram page gains stacked time-budget bars per entity, transition
  heatmap, bout table with CSV download.
- **Acceptance:** fixture tests with exact expected values; AppTest renders
  all three panels; CSV parses. Suite green.

## 6. [USER-SUPERVISED] Live mode + Hermes alerts — DO NOT run unattended

RTSP/webcam rolling analysis with event alerts through the user's Hermes
gateway. Requires cameras, service coordination, and GPU-budget decisions
an unattended agent must not make. Sketch: `src/live_runner.py` ring-buffer
ingestion → periodic mini-pipeline over the last N seconds → event diff →
alert hook (shell-out to `hermes` CLI). Design doc first, with the user.

---

## Done in v3 (for the record)

Drop→plan→progress UI, auto_profile, per-run Ask tab, summarizer with
LLM/heuristic paths + caching, resume-from-DB guards, cancel, ETA
calibration, dataset-exporter tab, per-frame transcript join, CPU smoke
harness. 68 tests.
