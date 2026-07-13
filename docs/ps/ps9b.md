# PS9b — Live captioning view + PS9 test suite (9B lane)

While a run is processing, the user should be able to *watch* the 9B
walk the video: the frame it's looking at, the prompt it was sent, the
caption it produced, the transcript around that moment, and 35B
verdicts landing in parallel. Plus the tests for the two-phase parallel
walker being built in `docs/ps/ps9a.md` (35B lane).

Code against the FROZEN CONTRACTS in `docs/ps/ps9a.md` — the storage
helpers (`get_latest_frame_captions`, `get_audio_segments_overlapping`,
`update_caption_change`, `insert_frame_caption(..., prompt=)`), the
`walker_concurrency` setting, and `DenseCaptionWalker(concurrency=)`
exist by the time both lanes merge. Do NOT reimplement them.

## Scope fence

- **Files you may create:** `ui/live_view.py`,
  `tests/test_dense_parallel.py`, `tests/test_live_view.py`
- **Files you may modify:** `ui/app.py` (progress screen only)
- **Do NOT touch:** `src/**` (35B lane owns it).

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.

## FROZEN CONTRACTS

### ui/live_view.py

```python
def fetch_live_state(db_path: Path, n: int = 5) -> Optional[dict]:
    """Pure data fetch — NO streamlit imports at module level needed
    for this function; testable headless.

    Opens the DB read-only: sqlite3.connect(f"file:{db_path}?mode=ro",
    uri=True). Missing file / missing table / empty table → None.

    Returns:
      {
        "current": {frame_idx, timestamp_ms, image_path, caption,
                    change_line, prompt, escalated},
        "recent":  [same-shape dicts, newest first, up to n],
        "transcript": [ {start_ms, end_ms, text} ... ]   # segments
                    overlapping [current.timestamp_ms - 1000,
                                 current.timestamp_ms + 1000]
        "verdicts": [ {frame_idx, verifier_caption, verifier_agrees,
                       verifier_model} ... ]  # last 5 verified rows,
                    newest verified_at first
        "counts": {"captioned": int, "escalated": int, "verified": int},
      }
    """

def render_live_view(db_path: Path) -> None:
    """Streamlit rendering of fetch_live_state: current frame image
    (st.image), caption + change line, an expander with the exact
    prompt, transcript ±1 s under the image, a second column streaming
    the last verifier verdicts (✓ agree / ✗ override + 35B caption),
    and a small trailing feed of the previous captions. Handles
    fetch_live_state(...) is None with st.caption("waiting for first
    caption…"). Never raises on a mid-write row.
    """
```

### ui/app.py wiring

On the progress screen, while the "Dense captions" stage is running
(or queued and the DB exists), render `render_live_view(db_path)`
inside `@st.fragment(run_every="1s")` so it refreshes without
rerunning the whole page or touching the pipeline thread. Read-only —
the live view must never open a writable connection to the run DB.
Degrade silently when `dense_captions` is off. Keep every existing
progress element (stage checklist, ETA) exactly as is.

## Tests

`tests/test_dense_parallel.py` — mock `requests.post` (thread-safe:
the walker calls it from `concurrency` worker threads; use a lock or
thread-safe side_effect). Real tmp-path `TempoGraphDB`s, no servers.

1. **Parity**: same 12-frame fixture run with `concurrency=1` and
   `concurrency=4` (deterministic mocked captions) → identical
   captions, identical escalated set after phase 2.
2. **Prompt persisted**: every `frame_captions` row has a non-empty
   `prompt`; in parallel mode it does NOT contain "Previous frame:".
3. **Ordering**: mocked completion order shuffled (delay by frame idx
   parity) → change lines still pair frame N with N-1 by `frame_idx`,
   not by completion order.
4. **Phase-2 failure tolerance**: change-request HTTP error on one
   pair → that row keeps `change_line=None`, escalation still computed
   from delta + jaccard, walk result counts the error, run completes.
5. **Resume**: pre-insert captions for half the frames → phase 1 only
   requests the missing half; phase 2 still covers all pairs.
6. **Cancel**: `cancel_event` set after N frames → walk returns
   cleanly, no hang (join with timeout in the test).
7. **Verifier transcript**: audio segments in the DB → the 35B payload
   prompt contains "Spoken audio" and the overlapping text; no
   segments → it doesn't. Overlap boundary: a segment ending exactly
   at `ts - 5000` is excluded, one straddling the edge is included.
8. **`get_audio_segments_overlapping`** boundary semantics directly
   (start-exclusive/end-exclusive per the contract).

`tests/test_live_view.py` — headless, no streamlit runtime needed
(test `fetch_live_state` only):

1. Missing DB file → None; empty `frame_captions` → None.
2. Populated fixture → correct `current` (highest `created_at`),
   `recent` ordering, transcript window ±1 s (segment overlapping the
   window included, distant segment excluded), verdict list, counts.
3. Read-only: after `fetch_live_state`, a writer connection can still
   INSERT immediately (no lingering lock).

## ACCEPTANCE (paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_dense_parallel.py tests/test_live_view.py -q
/home/ashie/anaconda3/bin/python3 -m pytest -q          # full suite green
timeout 12 /home/ashie/anaconda3/bin/python3 -m streamlit run ui/app.py \
  --server.headless true --server.port 8599 & sleep 8; curl -sf localhost:8599 >/dev/null && echo UI_BOOTS
/home/ashie/anaconda3/bin/python3 -m black ui/ tests/
git status --porcelain
```

(do not commit — gate reviewer merges both lanes)
