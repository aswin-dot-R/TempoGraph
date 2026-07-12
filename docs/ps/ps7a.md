# PS7a — Highlight reel: IMPLEMENTATION (35B lane)

"The most interesting 60 seconds." The frame scorer already ranks frames
by information density (`frames.delta_score`); stitch the top spans into
one summary clip. Tests in parallel (`docs/ps/ps7b.md`) against the
FROZEN CONTRACTS below.

## Scope fence

- **Files you may create:** `src/highlight_reel.py`
- **Files you may modify:** `ui/pages/Results.py`
- No test files — `tests/test_highlight_reel.py` belongs to the other lane.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.
`ffmpeg`/`ffprobe` are on PATH.

## FROZEN CONTRACTS

`src/highlight_reel.py` — pure span math separated from ffmpeg work
(mirror `src/clip_export.py`'s structure and reuse its ffmpeg helpers
where importable):

```python
def pick_highlight_spans(
    db_path: Path,
    target_duration_s: float = 60.0,
    min_gap_s: float = 3.0,
    span_padding_s: float = 1.5,
) -> List[Tuple[int, int]]:
    """Greedy: frames sorted by delta_score desc; accept a frame if its
    padded span (timestamp +/- span_padding_s, clamped >= 0) is >=
    min_gap_s away from every accepted span; stop when the summed span
    duration reaches target_duration_s or frames run out. Merge
    overlapping/touching accepted spans. Return [(start_ms, end_ms)]
    sorted by start. Empty DB -> []."""

def build_highlight_reel(
    video_path: Path,
    spans: List[Tuple[int, int]],
    out_path: Path,
    fade_s: float = 0.25,
) -> Path:
    """Cut each span (re-encode for frame accuracy), concatenate with
    xfade crossfades of fade_s (single span: no fade), write out_path,
    ffprobe-verify, return out_path. Raise RuntimeError with the ffmpeg
    stderr tail on failure. Empty spans -> ValueError."""
```

UI contract: a "Highlight Reel" section on the **Overview** tab —
target-duration slider (30/60/90 s, key `reel_duration`), a "Build reel"
button (key `reel_build`) that resolves the source video exactly like
the Clips section does (`run_meta.video_path`), writes
`<run_dir>/highlight_reel.mp4`, then `st.video` + a download button.
Existing reel file on disk renders immediately without rebuilding.
Missing source video → st.info, never a crash.

## ACCEPTANCE (paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest -q     # existing suite green
grep -q "highlight_reel" ui/pages/Results.py && echo WIRED
timeout 12 /home/ashie/anaconda3/bin/python3 -m streamlit run ui/app.py \
  --server.headless true --server.port 8599 & sleep 8; curl -sf localhost:8599 >/dev/null && echo UI_BOOTS
/home/ashie/anaconda3/bin/python3 -m black src/highlight_reel.py ui/pages/Results.py
git status --porcelain   # only the 2 scoped files
```

(do not commit — gate reviewer merges both lanes)
