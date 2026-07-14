# PS3 — EscalationVerifier: the 35B second-opinions in parallel

Step 3 of 4 of Dense Temporal Captioning
(design: `docs/superpowers/specs/2026-07-12-dense-temporal-captioning-design.md`).
Requires PS1 + PS2 merged. Paste every ACCEPTANCE output into your final
summary.

## Scope fence

- **Files you may modify:** `src/modules/dense_captioner.py`
- **Files you may create:** `tests/test_dense_verifier.py`
- Touch NOTHING else.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server, allocating GPU memory. Tests mock all HTTP. Interpreter:
`/home/ashie/anaconda3/bin/python3`.

## What you are building

The verifier runs **while the walker is still walking** — they write the
same DB concurrently from separate threads and separate connections (WAL
mode from PS1 makes this safe). The 35B lives at
`http://127.0.0.1:8096` — same OpenAI-compatible payload shape as PS2, but
it may receive TWO images (previous frame + current frame) in one message's
content list.

## Task 1 — EscalationVerifier class

In `src/modules/dense_captioner.py`:

```python
class EscalationVerifier:
    def __init__(
        self,
        db_path: Path,
        base_url: str = "http://127.0.0.1:8096",
        model_name: str = "ornith-1.0-35b",
        temperature: float = 0.1,
        max_tokens: int = 128,
        poll_interval_s: float = 2.0,
        batch_size: int = 8,
        request_timeout_s: float = 180.0,
        walker_done: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ): ...

    def run(self) -> dict:
        """Poll-verify loop. Returns {"verified": int, "agreed": int,
        "disagreed": int, "errors": int}.

        Loop: fetch_unverified_escalations(batch_size); for each row build
        the prompt (Task 2), call the 35B, parse, save_caption_verdict().
        When the fetch is empty: if walker_done is set (or was never
        provided), drain once more and exit; otherwise sleep
        poll_interval_s and poll again. Honor cancel_event between rows.
        """
```

`run()` opens its OWN `TempoGraphDB(db_path)`. Per-row HTTP failures are
logged, counted in `errors`, and the row is left unverified (it will be
retried on the next poll — cap retries per frame_idx at 3 in-memory, then
skip it for the rest of the run so the loop can terminate).

## Task 2 — verifier prompt + parser

Content list: text prompt, then previous frame image (when a `frames` row
with a smaller frame_idx exists), then current frame image.

```
A smaller vision model watched a video and flagged a big scene change here.
Its caption for THIS frame: {caption}
Its change note: {change_line}
You see the previous frame (first image, if present) and the current frame (last image).
Reply with EXACTLY two lines:
VERDICT: AGREE or DISAGREE — does its caption fairly describe the current frame?
CAPTION: your own one-sentence caption naming objects, attributes, surfaces, and background activity.
```

Parser `parse_verdict(text) -> Tuple[bool, str]` (module-level,
unit-testable): find `VERDICT:` line case-insensitively — `agrees=True`
iff it contains "agree" and not "disagree"; find `CAPTION:` line for
`verifier_caption`. Fallbacks: no VERDICT line → agrees=True (benefit of
the doubt); no CAPTION line → use the whole reply stripped, or
`"(no caption)"` if empty.

## Task 3 — the parallel orchestrator

Module-level function — this is the single entry point the pipeline (PS4)
will call:

```python
def run_dense_captioning(
    db_path: Path,
    walker_url: str = "http://127.0.0.1:8085",
    verifier_url: str = "http://127.0.0.1:8096",
    on_progress: Optional[Callable[[dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    **kwargs,
) -> dict:
    """Start EscalationVerifier.run() in a daemon thread, run
    DenseCaptionWalker.walk() in the calling thread, set the shared
    walker_done event, join the verifier thread (timeout 600 s), and
    return {"walker": <walk counts>, "verifier": <run counts>}.
    """
```

The verifier thread starts FIRST so verification overlaps the walk from the
first escalation onward. Progress events from both sides are forwarded to
`on_progress` with a `"who": "walker"|"verifier"` key added. If the walker
raises, set walker_done anyway and re-raise after joining the verifier.

## Task 4 — tests (`tests/test_dense_verifier.py`)

Same mocking approach as PS2 (monkeypatch `requests.post`; route by URL —
`:8085` → walker replies, `:8096` → verifier replies). Reuse/adapt the PS2
fixture (20 frames, tiny real JPEGs).

Cover:
1. `parse_verdict` — AGREE, DISAGREE, missing VERDICT fallback, missing
   CAPTION fallback.
2. Verifier alone on a pre-populated DB — 5 escalated rows → all verified,
   counts correct, `verified_at`/`verifier_model` set, loop exits because
   `walker_done` was pre-set.
3. Retry cap — mock raises forever for one frame_idx; verifier exits
   anyway, that row stays unverified, `errors >= 3`.
4. **True parallel run** — `run_dense_captioning` on the 20-frame fixture
   with walker mock captions crafted so ~6 frames escalate: afterwards
   every escalated row has a verdict, walker counts and verifier counts
   agree, and no "database is locked" ever surfaced. Assert (via a
   timestamp or ordering probe) that at least one verdict was written
   BEFORE the walker finished — proving overlap, not post-hoc verification.
5. Cancel — cancel_event set mid-run stops both threads cleanly; partial
   counts returned.

## ACCEPTANCE (run, paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest tests/test_dense_verifier.py -q
/home/ashie/anaconda3/bin/python3 -m pytest -q      # full suite green
git status --porcelain                               # only the 2 scoped files
```

## Commit message

```
Add EscalationVerifier and parallel run_dense_captioning orchestrator (PS3)
```
