# Dense Temporal Captioning — Design

Date: 2026-07-12. Status: approved by user (session with Fable as architect).
Executor: local Ornith models driven by opencode, one PS file at a time
(`docs/ps/ps1.md` … `ps4.md`). Fable writes specs and reviews diffs; it does
not write the feature code.

## Goal

Every analyzed video gets a dense, per-frame temporal record written by the
local vision models themselves:

- **Ornith 9B** (llama-server, `http://127.0.0.1:8085`, vision) — the
  **walker**. Walks the run's frames in order and writes two one-liners per
  frame into the DB: `caption` (what is in the frame) and `change_line`
  (what changed vs. the previous frame).
- **Ornith 35B** (llama-server, `http://127.0.0.1:8096`, vision) — the
  **verifier**. Runs **in parallel, while the walker is still writing**
  (they live on different GPUs). It consumes frames the walker flagged as
  big changes, re-examines them, and writes its own opinion + an
  agree/disagree verdict into the same table.

End state after a run: `frame_captions` is a complete temporal record —
"a big DB containing the temporal information" — which downstream
aggregation passes (and later NL search / highlight reel) consume.

## Why (research context)

Market research (2026-07-12) found no fully-local open-source tool doing
video → transcript + entities + events + graph + Q&A. The closest
architectural cousin (NVIDIA VSS blueprint) is built around dense VLM
captioning; adding it here removes the one architectural advantage VSS has
over TempoGraph while staying single-machine and `make install`-simple.

## What exists already (do not rebuild)

- `frames` table already stores every selected frame with `delta_score`
  (cheap "how much changed" signal from frame selection). The walker walks
  these rows; it does not decode the video itself.
- `LlamaServerBackend` already takes `base_url`; the OpenAI-compatible
  payload shape (base64 `image_url` items, `chat_template_kwargs`) is the
  reference for the walker/verifier HTTP calls.
- Resume guards (`run_stages`), `run_meta`, migration pattern in
  `TempoGraphDB._migrate()` (see `mask_rle`) — all reused.

## Architecture

New module `src/modules/dense_captioner.py`:

```
run_dense_captioning(db_path, walker_url, verifier_url, ...)
  ├─ EscalationVerifier thread  (own TempoGraphDB connection, polls DB)
  └─ DenseCaptionWalker loop    (own TempoGraphDB connection)
```

- **Walker** iterates `frames ORDER BY frame_idx`. For each frame: one HTTP
  call to the 9B with the frame image + the previous frame's caption in the
  prompt; parses `FRAME:` / `CHANGE:` lines; inserts a `frame_captions`
  row. Sets `escalated=1` when the change is big (see Escalation).
- **Verifier** polls `frame_captions WHERE escalated=1 AND verifier_agrees
  IS NULL`, sends the frame (plus previous frame when available) and the
  walker's two lines to the 35B, and UPDATEs the row with
  `verifier_caption`, `verifier_agrees`, `verifier_model`, `verified_at`.
  It starts before the walker and stops when the walker is done AND the
  escalation queue is drained.
- **Concurrent writes** are safe because `TempoGraphDB` enables WAL mode +
  `busy_timeout`, each thread owns its own connection, and both writers use
  short single-row transactions.

### Escalation trigger (two signals, either fires)

1. `delta_score` of the frame ≥ the run's 90th percentile (computed once at
   walk start — robust to the score's unit).
2. Token-set Jaccard similarity between the new caption and the previous
   caption < 0.3 (the scene *reads* different even if pixels moved little).

### Schema

```sql
CREATE TABLE IF NOT EXISTS frame_captions (
    frame_idx INTEGER PRIMARY KEY,
    caption TEXT NOT NULL,
    change_line TEXT,
    walker_model TEXT NOT NULL,
    escalated INTEGER NOT NULL DEFAULT 0,
    verifier_caption TEXT,
    verifier_agrees INTEGER,          -- NULL = not (yet) verified
    verifier_model TEXT,
    created_at TEXT NOT NULL,
    verified_at TEXT,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);
```

One row per frame; walker INSERTs, verifier UPDATEs — no cross-writer row
contention.

### Pipeline & UI integration

- New opt-in stage **"Dense captions"** in `PipelineV2.run()` directly after
  YOLO detection, with the standard `_stage` events, resume guard, and
  `mark_stage_complete`. Ctor knobs follow the existing `vlm_url` style:
  `dense_captions=False`, `walker_url`, `verifier_url`.
- Frame inspector shows caption + change line (+ verifier verdict when
  present) under each frame; Timeline gains dense-caption entries.
- `CaptionAggregator` gains a pass that loads the dense timeline and feeds
  a condensed version into aggregation; `analysis.json` gains
  `dense_timeline`.

### Adaptive density

The `frames` table is already delta-driven sampling (~1–2 fps typical), so
v1 captions exactly those frames. A future `dense_extra_fps` knob that
decodes extra frames inside high-delta windows is explicitly deferred
(YAGNI until the base layer proves out).

## Delivery plan

Four Ornith-sized PS files, each independently green:

| PS | Scope | New/changed |
|---|---|---|
| ps1 | Schema + WAL concurrency + DB helpers | `src/storage.py`, `tests/test_dense_schema.py` |
| ps2 | Walker (9B) with escalation + resume | `src/modules/dense_captioner.py`, `tests/test_dense_walker.py` |
| ps3 | Verifier (35B) + parallel orchestrator | same module, `tests/test_dense_verifier.py` |
| ps4 | Pipeline stage + UI + aggregator pass | `src/pipeline_v2.py`, `ui/pages/Results.py`, `src/aggregator.py`, tests |

Execution loop per PS: opencode session on Ornith 35B implements the PS →
Ornith 9B runs `docs/ps/review-checklist.md` → user brings diff + pasted
acceptance output back to Fable for gate review → commit with the message
given in the PS. `QUEUE.md` tracks state.

## Queue after this feature

click-to-play (PS exists) → NL search → highlight reel → open-source prep
(positioning: "open-source Twelve Labs on your own GPU"; GIF-first README;
Docker; VRAM table).

**NL search scope (user requirement, 2026-07-12):** the search index must
cover everything a human would look up, in one query surface — the Whisper
transcript (`audio_segments.text`), dense captions + change lines +
verifier opinions (`frame_captions`), detection class names
(`detections.class_name`), and aggregated entities/events from
`analysis.json`. Each hit carries `(run, timestamp_ms, source_type,
snippet)` so results click through to the exact moment regardless of which
modality matched.

Canonical example queries (user, 2026-07-12): *"show me the frame in which
keys are visible on a black table"*, *"play the part of the clip where dogs
are moving in the background"*. These are semantic, not keyword — so:

1. Walker captions must be **retrieval-grade**: name objects, their
   attributes, and surfaces/background ("keys on a black table", "two dogs
   moving in the background"), which the PS2 prompt enforces.
2. Search runs in two layers: FTS5 recall first, then an Ornith-assisted
   layer (query → search-term rewrite, and/or top-k rerank by the 9B) for
   phrasing gaps.
3. Result actions: "show frame" jumps the frame inspector; "play part"
   hands the matched span to the existing clip-export/player path
   (click-to-play integration).

## Testing philosophy

All unit/integration tests mock the HTTP layer (`requests.post`) — the
suite must pass with no llama-server running and no GPU. Each PS also
carries one optional live smoke command (skipped/absent in CI) the user can
run against the real servers. Interpreter for everything:
`/home/ashie/anaconda3/bin/python3`.
