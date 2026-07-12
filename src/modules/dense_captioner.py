"""DenseCaptionWalker: per-frame 9B vision captions with escalation.

Walks a run's ``frames`` table in order, calls the local Ornith 9B vision
server once per frame, writes one ``frame_captions`` row per frame, and
flags big scene changes (``escalated=1``) for the 35B verifier (PS3).

All HTTP is via the OpenAI-compatible ``/v1/chat/completions`` endpoint on
a llama-server (default ``http://127.0.0.1:8085``).  Tests mock ``requests``
so the suite runs with no server running.
"""

from __future__ import annotations

import base64
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

from src.storage import TempoGraphDB

_logger = logging.getLogger(__name__)

# ── prompt template ──────────────────────────────────────────────────

_FRAME_PROMPT = (
    "You are watching a video one frame at a time.\n"
    "Previous frame: {prev_caption}\n"
    "Reply with EXACTLY two lines and nothing else:\n"
    "FRAME: one sentence naming the visible objects with their "
    "colors/attributes, the surface or setting they are on, and anything "
    "happening in the background (e.g. "
    '"keys and a phone on a black table, two dogs moving in the background").\n'
    "CHANGE: one short sentence on what changed since the previous frame, "
    'or "no change".\n'
)


# ── Task 2: prompt + parser ────────────────────────────────────────


def parse_two_lines(text: str) -> Tuple[str, Optional[str]]:
    """Extract ``FRAME:`` and ``CHANGE:`` lines from a model reply.

    Search is case-insensitive for the ``FRAME:`` / ``CHANGE:`` prefixes.
    Falls back gracefully when one or both markers are missing.

    Args:
        text: The raw reply text from the 9B server.

    Returns:
        ``(caption, change_line)`` — ``change_line`` is ``None`` if no
        ``CHANGE:`` line was found.  An empty reply yields the caption
        ``"(no caption)"`` and ``change_line=None``.
    """
    if not text or not text.strip():
        return "(no caption)", None

    caption: Optional[str] = None
    change_line: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # ``FRAME:`` prefix (case-insensitive)
        m = re.match(r"^[Ff][Rr][Aa][Mm][Ee]\s*:\s*(.+)$", line)
        if m:
            caption = m.group(1).strip()
            continue
        # ``CHANGE:`` prefix (case-insensitive)
        m = re.match(r"^[Cc][Hh][Aa][Nn][Gg][Ee]\s*:\s*(.+)$", line)
        if m:
            change_line = m.group(1).strip()

    # Fallback: no FRAME: prefix → use the first non-empty line.
    if caption is None:
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped:
                caption = stripped
                break
        if caption is None:
            caption = "(no caption)"

    return caption, change_line


# ── Task 3: escalation logic ───────────────────────────────────────


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two lowercased captions.

    Empty ∪ empty → 1.0.

    Args:
        a: First caption string.
        b: Second caption string.

    Returns:
        Float in [0.0, 1.0].
    """
    tokens_a = set(a.lower().split()) if a else set()
    tokens_b = set(b.lower().split()) if b else set()
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def should_escalate(
    delta_score: float,
    delta_threshold: float,
    caption: str,
    prev_caption: Optional[str],
    similarity_floor: float,
) -> bool:
    """True if this frame should be flagged as a big scene change.

    Escalation fires when *either* signal triggers:

    1. ``delta_score >= delta_threshold`` (pixel-level change is big), or
    2. ``prev_caption`` is not None **and**
       ``jaccard(caption, prev_caption) < similarity_floor``
       (the caption *reads* different).

    When ``prev_caption`` is None (first frame) only the delta signal can
    fire — the similarity signal is undefined.

    Args:
        delta_score: The frame's delta_score from the ``frames`` table.
        delta_threshold: Pre-computed threshold (e.g. 90th percentile).
        caption: Parsed caption for this frame.
        prev_caption: Parsed caption from the previous frame, or None.
        similarity_floor: Threshold below which captions are "different".

    Returns:
        True if this frame should be escalated for verifier review.
    """
    if delta_score >= delta_threshold:
        return True
    if prev_caption is not None and jaccard(caption, prev_caption) < similarity_floor:
        return True
    return False


def _percentile(scores: List[float], pct: float) -> float:
    """Pure-Python percentile (linear interpolation, like numpy default).

    Args:
        scores: Non-empty list of numeric values.
        pct: Percentile in [0, 100].

    Returns:
        The ``pct``-th percentile value.
    """
    if not scores:
        return 0.0
    s = sorted(scores)
    if len(s) < 2:
        return s[0]
    pos = (pct / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ── Task 1, 4: DenseCaptionWalker ──────────────────────────────────


class DenseCaptionWalker:
    """Iterate a run's frames, caption each with the 9B vision server.

    Opens its own :class:`src.storage.TempoGraphDB` (WAL handles the
    concurrent verifier in PS3).  Never crashes on a single HTTP failure —
    logs and counts it, moves on.
    """

    def __init__(
        self,
        db_path: Path,
        base_url: str = "http://127.0.0.1:8085",
        model_name: str = "ornith-1.0-9b",
        temperature: float = 0.1,
        max_tokens: int = 96,
        escalation_percentile: float = 90.0,
        caption_similarity_floor: float = 0.3,
        request_timeout_s: float = 120.0,
        on_progress: Optional[Callable[[dict], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """
        Args:
            db_path: Path to the run's ``tempograph.db``.
            base_url: llama-server base URL (must expose
                ``/v1/chat/completions``).
            model_name: Model name sent in the request payload.
            temperature: Sampling temperature for the 9B.
            max_tokens: Max tokens per response.
            escalation_percentile: Percentile of ``frames.delta_score``
                used as the escalation threshold.
            caption_similarity_floor: Jaccard floor below which a caption
                change triggers escalation.
            request_timeout_s: HTTP timeout per frame.
            on_progress: Callback invoked after each frame with a dict
                ``{frame_idx, done, total, escalated}``.
            cancel_event: If set, ``walk()`` stops cleanly after the
                current frame.
        """
        self.db_path = Path(db_path)
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.escalation_percentile = escalation_percentile
        self.caption_similarity_floor = caption_similarity_floor
        self.request_timeout_s = request_timeout_s
        self.on_progress = on_progress
        self.cancel_event = cancel_event
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    # ── public API ─────────────────────────────────────────────────

    def walk(self) -> Dict[str, int]:
        """Caption every frame not yet in ``frame_captions``.

        Returns:
            ``{"captioned": int, "escalated": int, "skipped": int,
            "errors": int}``.
        """
        db = TempoGraphDB(self.db_path)
        try:
            return self._walk(db)
        finally:
            db.close()

    # ── internals ──────────────────────────────────────────────────

    def _walk(self, db: TempoGraphDB) -> Dict[str, int]:
        frame_indices = db.get_all_frame_indices()
        total = len(frame_indices)
        if total == 0:
            self.logger.info("No frames to caption.")
            return {"captioned": 0, "escalated": 0, "skipped": 0, "errors": 0}

        # Compute delta_threshold ONCE from the frames table.
        raw_scores = [
            row["delta_score"]
            for row in db._conn.execute(
                "SELECT delta_score FROM frames ORDER BY frame_idx"
            ).fetchall()
        ]
        if len(raw_scores) < 10:
            delta_threshold = max(raw_scores) if raw_scores else 0.0
        else:
            delta_threshold = _percentile(raw_scores, self.escalation_percentile)
        self.logger.info(f"Walk: {total} frames, delta_threshold={delta_threshold:.4f}")

        # Set of already-captioned frame indices (for resume).
        existing = {
            row["frame_idx"]
            for row in db._conn.execute(
                "SELECT frame_idx FROM frame_captions"
            ).fetchall()
        }

        captioned = 0
        escalated = 0
        skipped = 0
        errors = 0
        prev_caption: Optional[str] = None

        for frame_idx in frame_indices:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.logger.info(
                    f"Cancel requested after {captioned} captioned frames."
                )
                break

            # Resume: skip already-captioned frames.
            if frame_idx in existing:
                skipped += 1
                # Still need prev_caption for prompt continuity even on
                # skipped frames — read it from the DB.
                row = db.get_frame_caption(frame_idx)
                if row and row.get("caption"):
                    prev_caption = row["caption"]
                if self.on_progress is not None:
                    try:
                        self.on_progress(
                            {
                                "frame_idx": frame_idx,
                                "done": captioned + skipped,
                                "total": total,
                                "escalated": escalated,
                            }
                        )
                    except Exception:
                        pass
                continue

            # Encode image.
            frame = db.get_frame(frame_idx)
            if not frame:
                self.logger.warning(
                    f"Frame {frame_idx} disappeared between query and read."
                )
                errors += 1
                if self.on_progress is not None:
                    try:
                        self.on_progress(
                            {
                                "frame_idx": frame_idx,
                                "done": captioned + skipped,
                                "total": total,
                                "escalated": escalated,
                            }
                        )
                    except Exception:
                        pass
                continue

            try:
                b64 = self._encode_image(Path(frame["image_path"]))
            except OSError as e:
                self.logger.error(f"Frame {frame_idx} image unreadable: {e}")
                errors += 1
                continue

            # Build prompt.
            prev = prev_caption if prev_caption is not None else "(first frame)"
            prompt = _FRAME_PROMPT.format(prev_caption=prev)

            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
            }

            # HTTP call — never crash the walk on failure.
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.request_timeout_s,
                )
                resp.raise_for_status()
                raw = resp.json()
                reply = (
                    raw.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
            except Exception as e:
                self.logger.error(f"Frame {frame_idx} HTTP failed: {e}")
                errors += 1
                if self.on_progress is not None:
                    try:
                        self.on_progress(
                            {
                                "frame_idx": frame_idx,
                                "done": captioned + skipped,
                                "total": total,
                                "escalated": escalated,
                            }
                        )
                    except Exception:
                        pass
                # On error: no caption to store, prev stays as-is so the
                # next frame's prompt still has *something*.
                continue

            caption, change_line = parse_two_lines(reply)
            escalated_flag = should_escalate(
                delta_score=frame["delta_score"],
                delta_threshold=delta_threshold,
                caption=caption,
                prev_caption=prev_caption,
                similarity_floor=self.caption_similarity_floor,
            )
            db.insert_frame_caption(
                frame_idx=frame_idx,
                caption=caption,
                change_line=change_line,
                walker_model=self.model_name,
                escalated=escalated_flag,
            )
            captioned += 1
            if escalated_flag:
                escalated += 1
                self.logger.debug(
                    f"Frame {frame_idx} escalated "
                    f"(delta={frame['delta_score']:.4f}, "
                    f"threshold={delta_threshold:.4f})"
                )

            prev_caption = caption

            if self.on_progress is not None:
                try:
                    self.on_progress(
                        {
                            "frame_idx": frame_idx,
                            "done": captioned + skipped,
                            "total": total,
                            "escalated": escalated,
                        }
                    )
                except Exception:
                    pass

        self.logger.info(
            f"Walk complete: captioned={captioned}, escalated={escalated}, "
            f"skipped={skipped}, errors={errors}"
        )
        return {
            "captioned": captioned,
            "escalated": escalated,
            "skipped": skipped,
            "errors": errors,
        }

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════
# PS3 — EscalationVerifier: the 35B second-opinions in parallel
# ═══════════════════════════════════════════════════════════════════

# ── verifier prompt template ──────────────────────────────────────

_VERDICT_PROMPT = (
    "A smaller vision model watched a video and flagged a big scene change "
    "here.\n"
    "Its caption for THIS frame: {caption}\n"
    "Its change note: {change_line}\n"
    "You see the previous frame (first image, if present) and the current "
    "frame (last image).\n"
    "Reply with EXACTLY two lines:\n"
    "VERDICT: AGREE or DISAGREE — does its caption fairly describe the "
    "current frame?\n"
    "CAPTION: your own one-sentence caption naming objects, attributes, "
    "surfaces, and background activity.\n"
)


# ── Task 2: verifier prompt + parser ──────────────────────────────


def parse_verdict(text: str) -> Tuple[bool, str]:
    """Parse the 35B model's ``VERDICT:`` / ``CAPTION:`` reply.

    Searches for a line containing ``VERDICT:`` (case-insensitive).
    ``agrees`` is ``True`` iff that line contains ``agree`` but not
    ``disagree``.  Searches for a line containing ``CAPTION:`` for the
    verifier caption.

    Fallbacks:
    - No ``VERDICT:`` line → ``agrees=True`` (benefit of the doubt).
    - No ``CAPTION:`` line → whole reply stripped, or ``"(no caption)"``
      if the reply is empty.

    Args:
        text: Raw reply text from the 35B model.

    Returns:
        ``(agrees, verifier_caption)``.
    """
    text = text.strip()
    agrees = True  # fallback: benefit of the doubt
    caption: Optional[str] = None

    # First pass: look for the VERDICT line.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "verdict:" in line.lower():
            agrees = "agree" in line.lower() and "disagree" not in line.lower()
            break  # take the first VERDICT line

    # Second pass: look for the CAPTION line.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "caption:" in line.lower():
            # Everything after the first ":" on that line.
            after = line.split(":", 1)[1].strip()
            if after:
                caption = after
            break  # take the first CAPTION line

    # Fallback for missing CAPTION.
    if caption is None:
        if text:
            caption = text
        else:
            caption = "(no caption)"

    return agrees, caption


# ── Task 1: EscalationVerifier class ──────────────────────────────


class EscalationVerifier:
    """Poll-verify loop for escalated frames.

    Runs on a daemon thread while :class:`DenseCaptionWalker` walks in the
    calling thread.  Opens its own :class:`src.storage.TempoGraphDB`
    connection — WAL mode makes concurrent writes safe.

    Per-row HTTP failures are logged, counted in ``errors``, and the row is
    left unverified (retried on the next poll).  In-memory retry cap of 3
    per ``frame_idx``; after that the row is skipped for the rest of the
    run so the loop can still terminate.
    """

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
        on_progress: Optional[Callable[[dict], None]] = None,
    ):
        """
        Args:
            db_path: Path to the run's ``tempograph.db``.
            base_url: 35B llama-server base URL (must expose
                ``/v1/chat/completions``).
            model_name: Model name sent in the request payload.
            temperature: Sampling temperature for the 35B.
            max_tokens: Max tokens per response.
            poll_interval_s: Seconds to sleep when no escalations are
                pending and the walker is still walking.
            batch_size: Max unverified rows to fetch per poll.
            request_timeout_s: HTTP timeout per request.
            walker_done: Set when the walker has finished.  When this is set
                (or was never provided) the verifier drains one last time
                and exits.
            cancel_event: If set, ``run()`` stops cleanly after the current
                row.
            on_progress: Callback invoked periodically with progress dicts.
        """
        self.db_path = Path(db_path)
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.poll_interval_s = poll_interval_s
        self.batch_size = batch_size
        self.request_timeout_s = request_timeout_s
        self.walker_done = walker_done
        self.cancel_event = cancel_event
        self.on_progress = on_progress
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    # ── public API ─────────────────────────────────────────────────

    def run(self) -> Dict[str, int]:
        """Poll-verify loop.

        Returns:
            ``{"verified": int, "agreed": int, "disagreed": int,
            "errors": int}``.
        """
        db = TempoGraphDB(self.db_path)
        try:
            return self._run(db)
        finally:
            db.close()

    # ── internals ──────────────────────────────────────────────────

    def _run(self, db: TempoGraphDB) -> Dict[str, int]:
        verified = 0
        agreed = 0
        disagreed = 0
        errors = 0
        retry_counts: Dict[int, int] = {}  # frame_idx → attempts

        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.logger.info(f"Cancel requested after {verified} verified rows.")
                break

            # Capped rows stay unverified in the DB, so they would shadow
            # later rows behind a plain LIMIT batch_size — over-fetch by
            # the number of capped frames and filter them out here.
            capped = {i for i, n in retry_counts.items() if n >= 3}
            fetched = db.fetch_unverified_escalations(self.batch_size + len(capped))
            rows = [r for r in fetched if r["frame_idx"] not in capped][
                : self.batch_size
            ]
            self.logger.debug(
                f"Fetch returned {len(fetched)} rows "
                f"({len(rows)} eligible, {len(capped)} capped)"
            )

            if not rows:
                # Nothing eligible right now. If the walker is done (or was
                # never provided), drain once more to catch escalations
                # written in the final moments, then exit. Otherwise keep
                # polling — new escalations may still arrive.
                if self.walker_done is None or self.walker_done.is_set():
                    drain_rows = [
                        r
                        for r in db.fetch_unverified_escalations(10_000_000)
                        if retry_counts.get(r["frame_idx"], 0) < 3
                    ]
                    for row in drain_rows:
                        if self.cancel_event is not None and self.cancel_event.is_set():
                            break
                        result = self._process_row(db, row, retry_counts)
                        if result is None:
                            errors += 1
                        else:
                            verified += 1
                            if result:
                                agreed += 1
                            else:
                                disagreed += 1
                    # Drain is done — exit the loop regardless of whether
                    # we got rows in the drain pass.
                    break
                else:
                    time.sleep(self.poll_interval_s)
                    continue

            for row in rows:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    break
                frame_idx = row["frame_idx"]
                result = self._process_row(db, row, retry_counts)
                if result is None:
                    # _process_row failed (HTTP error, image missing, etc.)
                    errors += 1
                else:
                    verified += 1
                    if result:
                        agreed += 1
                    else:
                        disagreed += 1
                    if self.on_progress is not None:
                        try:
                            self.on_progress(
                                {
                                    "verified": verified,
                                    "agreed": agreed,
                                    "disagreed": disagreed,
                                    "errors": errors,
                                    "frame_idx": frame_idx,
                                    "time": time.time(),
                                }
                            )
                        except Exception:
                            pass

        self.logger.info(
            f"Verify complete: verified={verified}, agreed={agreed}, "
            f"disagreed={disagreed}, errors={errors}"
        )
        return {
            "verified": verified,
            "agreed": agreed,
            "disagreed": disagreed,
            "errors": errors,
        }

    def _process_row(
        self,
        db: TempoGraphDB,
        row,
        retry_counts: Dict[int, int],
    ) -> Optional[bool]:
        """Process a single escalation row.

        Returns ``True`` if the verifier agreed, ``False`` if it disagreed,
        or ``None`` on any failure (row left unverified, retry counted).
        """
        frame_idx = row["frame_idx"]
        self.logger.info(f"Processing frame {frame_idx}")

        # Skip if we've hit the retry cap for this frame.
        if retry_counts.get(frame_idx, 0) >= 3:
            self.logger.warning(
                f"Frame {frame_idx}: retry cap reached "
                f"({retry_counts[frame_idx]}), skipping."
            )
            return None

        # Read the current frame image.
        current_frame = db.get_frame(frame_idx)
        if not current_frame:
            self.logger.warning(
                f"Frame {frame_idx} disappeared between query and read."
            )
            retry_counts[frame_idx] = retry_counts.get(frame_idx, 0) + 1
            return None

        try:
            current_b64 = self._encode_image(Path(current_frame["image_path"]))
        except OSError as e:
            self.logger.error(f"Frame {frame_idx} image unreadable: {e}")
            retry_counts[frame_idx] = retry_counts.get(frame_idx, 0) + 1
            return None

        # Look for the previous frame (largest frame_idx < current).
        prev_b64: Optional[str] = None
        prev_frame = db._conn.execute(
            "SELECT * FROM frames WHERE frame_idx < ? "
            "ORDER BY frame_idx DESC LIMIT 1",
            (frame_idx,),
        ).fetchone()
        if prev_frame:
            try:
                prev_b64 = self._encode_image(Path(prev_frame["image_path"]))
            except OSError as e:
                self.logger.warning(f"Frame {frame_idx} prev image unreadable: {e}")

        # Build prompt.
        caption = row["caption"]
        change_line = row["change_line"] or ""
        prompt = _VERDICT_PROMPT.format(caption=caption, change_line=change_line)

        # Build content list: text + optional prev image + current image.
        content: list = [{"type": "text", "text": prompt}]
        if prev_b64 is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{prev_b64}"},
                }
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{current_b64}"},
            }
        )

        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

        # HTTP call — never crash the loop on failure.
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
            raw = resp.json()
            reply = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            self.logger.error(f"Frame {frame_idx} HTTP failed: {e}")
            retry_counts[frame_idx] = retry_counts.get(frame_idx, 0) + 1
            return None

        # Parse verdict + caption.
        agrees, verifier_caption = parse_verdict(reply)

        # Persist the verdict.
        try:
            db.save_caption_verdict(
                frame_idx=frame_idx,
                verifier_caption=verifier_caption,
                verifier_agrees=agrees,
                verifier_model=self.model_name,
            )
            # Clear retry count on success.
            retry_counts.pop(frame_idx, None)
            return agrees
        except Exception as e:
            self.logger.error(f"Frame {frame_idx} save_caption_verdict failed: {e}")
            retry_counts[frame_idx] = retry_counts.get(frame_idx, 0) + 1
            return None

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# ── Task 3: parallel orchestrator ─────────────────────────────────


def run_dense_captioning(
    db_path: Path,
    walker_url: str = "http://127.0.0.1:8085",
    verifier_url: str = "http://127.0.0.1:8096",
    on_progress: Optional[Callable[[dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    **kwargs,
) -> Dict[str, Dict[str, int]]:
    """Run the walker and verifier in parallel.

    Starts :class:`EscalationVerifier` on a daemon thread (so it can verify
    escalations as they appear) and runs :meth:`DenseCaptionWalker.walk` in
    the calling thread.  Sets the shared ``walker_done`` event when the
    walker finishes, joins the verifier thread (timeout 600 s), and returns
    combined counts.

    Progress events from both sides are forwarded to ``on_progress`` with
    a ``"who"`` key added (``"walker"`` or ``"verifier"``).

    If the walker raises, ``walker_done`` is set anyway and the verifier
    thread is joined before re-raising.

    Args:
        db_path: Path to the run's ``tempograph.db``.
        walker_url: URL of the 9B llama-server.
        verifier_url: URL of the 35B llama-server.
        on_progress: Callback for progress events.
        cancel_event: If set, stops both threads.
        **kwargs: Forwarded to both :class:`DenseCaptionWalker` and
            :class:`EscalationVerifier` constructors.

    Returns:
        ``{"walker": <walker counts>, "verifier": <verifier counts>}``.
    """
    walker_done = threading.Event()

    # Capture verifier's return value via a shared dict.
    verifier_run_result: Dict[str, int] = {}

    def _verifier_wrapper():
        import logging

        logging.getLogger(__name__).info("Verifier thread starting")
        verifier_run_result.update(verifier.run())
        logging.getLogger(__name__).info("Verifier thread finished")

    # ── separate kwargs for each worker ──────────────────────────

    # Verifier-specific kwargs that the walker doesn't accept.
    _VERIFIER_ONLY = {
        "poll_interval_s",
        "batch_size",
        "walker_done",  # not a real kwarg but kept for clarity
    }
    verifier_kwargs = {k: v for k, v in kwargs.items() if k in _VERIFIER_ONLY}
    walker_kwargs = {k: v for k, v in kwargs.items() if k not in _VERIFIER_ONLY}

    # ── progress forwarder (adds "who" key) ──────────────────────

    def _forward(who: str, data: dict) -> None:
        if on_progress is not None:
            try:
                event = {**data, "who": who}
                on_progress(event)
            except Exception:
                pass

    # ── instantiate both workers ─────────────────────────────────

    walker = DenseCaptionWalker(
        db_path=db_path,
        base_url=walker_url,
        on_progress=lambda d: _forward("walker", d),
        cancel_event=cancel_event,
        **walker_kwargs,
    )
    verifier = EscalationVerifier(
        db_path=db_path,
        base_url=verifier_url,
        walker_done=walker_done,
        cancel_event=cancel_event,
        on_progress=lambda d: _forward("verifier", d),
        **verifier_kwargs,
    )

    # ── start verifier on daemon thread FIRST ────────────────────

    verifier_thread = threading.Thread(
        target=_verifier_wrapper,
        daemon=True,
        name="DenseCaptionVerifier",
    )
    verifier_thread.start()

    # ── run walker in the calling thread ─────────────────────────

    walker_result: Dict[str, int] = {}

    try:
        walker_result = walker.walk()
    except Exception as e:
        _logger.error(f"Walker raised: {e}")
        walker_done.set()
        verifier_thread.join(timeout=600)
        raise

    # ── walker done — signal verifier and join ───────────────────

    walker_done.set()
    verifier_thread.join(timeout=600)

    return {"walker": walker_result, "verifier": verifier_run_result}
