# PS7b — Highlight reel: TEST SUITE (9B lane)

Tests for `src/highlight_reel.py`, written in parallel with the
implementation (`docs/ps/ps7a.md`) against its FROZEN CONTRACTS. Do not
read/wait for the implementation. Top of file:
`pytest.importorskip("src.highlight_reel")`.

## Scope fence

- **Files you may create:** `tests/test_highlight_reel.py`
- Touch NOTHING else.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.

## Fixture recipe

- DB: `TempoGraphDB` with hand-placed frames — you control timestamps
  and delta_scores exactly, so expected spans are computable by hand.
- Video: 14 s ffmpeg testsrc clip, generated the way
  `tests/test_clip_export.py` does; verify durations with ffprobe the
  same way.

## Tests

1. **Greedy pick, hand-checked** — 10 frames at 1 s intervals with
   delta_scores [9,1,8,1,7,...]: with target 6 s, padding 1.5 s, min_gap
   3 s, assert the exact expected span list (compute it in a comment).
2. **Min-gap respected** — two top frames 1 s apart → only one span
   accepted (the higher-scored one).
3. **Merge overlapping** — top frames 2 s apart (padded spans overlap)
   in the SAME accepted window → merged into one span, not two.
4. **Target duration stops accumulation** — total accepted span time
   ≤ target + one span's worth (greedy overshoot bound).
5. **Empty DB → []** and single-frame DB → one clamped span.
6. **ffmpeg integration** — real testsrc video + 2 hand spans →
   `build_highlight_reel` output exists, ffprobe duration ≈ sum of span
   durations − fade_s (±0.5 s tolerance), h264 stream present.
7. **Single span, no fade** — output duration ≈ span duration.
8. **Empty spans → ValueError**; nonexistent video → RuntimeError.

## ACCEPTANCE (paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_highlight_reel.py --collect-only -q
/home/ashie/anaconda3/bin/python3 -m black tests/test_highlight_reel.py
git status --porcelain   # only tests/test_highlight_reel.py
```

(do not commit — gate reviewer merges both lanes)
