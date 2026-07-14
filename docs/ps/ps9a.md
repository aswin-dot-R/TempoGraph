# PS9a — Parallel dense captioning + transcript-aware verifier (35B lane)

The dense-caption walker is the pipeline's slowest stage: one frame at a
time, one HTTP request at a time, because each frame's prompt embeds the
*previous* frame's caption. Split it into two phases so captions run
concurrently, persist the exact prompt per row, and give the 35B
verifier the transcript around each escalated frame. UI + tests run in
parallel (`docs/ps/ps9b.md` — the 9B lane).

## Scope fence

- **Files you may modify:** `src/storage.py`,
  `src/modules/dense_captioner.py`, `src/settings.py`,
  `src/pipeline_v2.py`
- **Do NOT touch:** `ui/**`, `tests/**` (9B lane owns those).
- No test files.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.
Stdlib + `requests` only — no aiohttp/httpx; use
`concurrent.futures.ThreadPoolExecutor`.

## FROZEN CONTRACTS (the 9B lane codes against these — do not deviate)

### storage.py

```python
# frame_captions gains one column: prompt TEXT (nullable).
# Migrate existing DBs: PRAGMA table_info check + ALTER TABLE ADD COLUMN
# in __init__ (same pattern as any existing column migration).

def insert_frame_caption(
    self, frame_idx, caption, change_line, walker_model,
    escalated: bool = False, prompt: Optional[str] = None,
) -> None: ...          # existing callers keep working (kwarg optional)

def update_caption_change(
    self, frame_idx: int, change_line: Optional[str], escalated: bool,
) -> None: ...          # phase-2 UPDATE of change_line + escalated. Commits.

def get_latest_frame_captions(self, limit: int = 5) -> list:
    """Most recent walker rows, newest first, JOINed to frames so each
    row also has image_path, timestamp_ms, delta_score."""

def get_audio_segments_overlapping(self, start_ms: int, end_ms: int) -> list:
    """Segments where seg.start_ms < end_ms AND seg.end_ms > start_ms,
    ordered by start_ms."""
```

### settings.py

```python
walker_concurrency: int   # TEMPOGRAPH_WALKER_CONCURRENCY, default 4
```

(invalid/non-int env value → fall back to 4, don't raise)

### dense_captioner.py

```python
DenseCaptionWalker(..., concurrency: int | None = None)
# None → get_settings().walker_concurrency. 1 → exact current
# sequential behaviour (byte-for-byte same prompts as today).

run_dense_captioning(..., concurrency passed through via **kwargs
# (walker-only kwarg — add to the kwarg-split so the verifier
# never receives it)
```

Two-phase walk (when `concurrency > 1`):

1. **Phase 1 — parallel captions.** Per-frame prompt is a
   parallel-safe variant of `_FRAME_PROMPT` **without** the
   `Previous frame:` line and **without** the CHANGE instruction —
   ask for the single `FRAME:` line only (`max_tokens` can stay).
   Submit frames to a `ThreadPoolExecutor(max_workers=concurrency)`;
   each worker POSTs to the walker URL exactly as today (llama-server
   continuous batching does the rest — works even if the server has
   `-np 1`, it just queues). Each completed frame is written
   immediately via `insert_frame_caption(..., escalated=False,
   change_line=None, prompt=<exact rendered prompt>)`. Per-frame HTTP
   errors: log, count, continue (same tolerance as today). Respect
   `cancel_event` between submissions and skip already-captioned
   frames (resume unchanged).
2. **Phase 2 — change lines + escalation.** After phase 1, walk the
   captioned frames in `frame_idx` order. For each consecutive pair,
   one **text-only** request to the walker URL: given caption N-1 and
   caption N, reply `CHANGE: <one short sentence>` or
   `CHANGE: no change` (`max_tokens=48`). Parse with the existing
   `parse_two_lines` fallback rules. Compute `escalated` with the
   existing `should_escalate(delta_score, delta_threshold, caption,
   prev_caption, similarity_floor)` — identical thresholds and
   percentile logic as today. Persist via `update_caption_change`.
   First frame: `change_line=None`, delta-only escalation. Phase 2
   requests may also run through the same executor (inputs are fixed
   text), but rows must be UPDATEd regardless of completion order.
   HTTP failure on a pair → `change_line=None`, escalation from
   delta + jaccard only (never lose the caption).
3. `on_progress` events keep the existing dict shape
   `{frame_idx, done, total, escalated}`; add `"phase": 1 | 2`.
   `walker_done` is set only after phase 2 finishes (verifier keeps
   polling until escalations are final).
4. Sequential mode (`concurrency == 1`) must also persist `prompt=`
   on every row — that's the only change to the sequential path.

### Verifier: transcript context

In `EscalationVerifier._process_row`, look up the frame's
`timestamp_ms`, fetch `get_audio_segments_overlapping(ts - 5000,
ts + 5000)`, and when non-empty append to the prompt:

```
Spoken audio around this moment (±5 s):
"<seg text>" (t=12.3s)
...
```

No segments (or no audio stage) → prompt unchanged. Cap at 6 segments.

### pipeline_v2.py

Thread a `walker_concurrency` ctor kwarg (default `None` →
settings) through to `run_dense_captioning`. Nothing else changes.

## ACCEPTANCE (paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest -q                      # suite green (existing tests untouched)
/home/ashie/anaconda3/bin/python3 - <<'EOF'                          # migration adds prompt column
import sqlite3, tempfile, os
from src.storage import TempoGraphDB
p = tempfile.mktemp(suffix=".db")
db = TempoGraphDB(p); cols = [r[1] for r in db._conn.execute("PRAGMA table_info(frame_captions)")]
assert "prompt" in cols, cols; print("PROMPT_COL_OK")
EOF
TEMPOGRAPH_WALKER_CONCURRENCY=9 /home/ashie/anaconda3/bin/python3 -c "
from src.settings import get_settings; print(get_settings().walker_concurrency)"   # 9
/home/ashie/anaconda3/bin/python3 -m black src/
git status --porcelain
```

(do not commit — gate reviewer merges both lanes)
