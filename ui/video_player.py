"""Streamlit-free video player helpers for click-to-play on the Results page.

All logic here is pure Python except ``render_player``, which emits Streamlit
widgets. Everything else (``resolve_video``, ``make_strip_mapper``) is testable
without Streamlit.
"""

from __future__ import annotations

import cv2
import streamlit as st
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class VideoSource:
    """A video file plus the time mapping to use when seeking.

    Attributes:
        path: Path to the video file to play.
        kind: ``"original"`` for the source video, ``"strip"`` for an
            annotated strip.
        time_mapper: Callable that converts source seconds (used by the
            timestamp buttons) into playback seconds on ``path``.
    """

    path: Path
    kind: str
    time_mapper: Callable[[float], float]


def make_strip_mapper(
    source_dur_s: float, strip_dur_s: float
) -> Callable[[float], float]:
    """Return a proportional time mapper from source seconds -> strip seconds.

    The mapper scales linearly: ``t * (strip_dur / source_dur)``, clamped to
    ``[0, strip_dur_s]``. When ``source_dur_s <= 0`` (unknown or empty source),
    the mapper always returns ``0.0``.

    Args:
        source_dur_s: Duration of the original source video in seconds.
        strip_dur_s: Duration of the annotated strip in seconds.

    Returns:
        A callable mapping source seconds to strip seconds.
    """

    def _mapper(t: float) -> float:
        if source_dur_s <= 0:
            return 0.0
        ratio = strip_dur_s / source_dur_s
        mapped = t * ratio
        if mapped < 0:
            return 0.0
        if mapped > strip_dur_s:
            return strip_dur_s
        return mapped

    return _mapper


def _probe_strip_duration(video_path: Path) -> float:
    """Return the duration in seconds of *video_path* using cv2.

    Falls back to ``4.0`` fps when ``CAP_PROP_FPS`` is zero/None, mirroring the
    pattern already used in ``ui/pages/Results.py``.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if fps > 0 and n_frames:
                return n_frames / fps
        finally:
            cap.release()
    except Exception:
        pass
    return 0.0


def _source_duration_from_db(db) -> float:
    """Return the source video duration in seconds using the db handle.

    Looks up ``max(timestamp_ms) / 1000`` across the frames table. Returns
    ``0.0`` if the query fails or the table is empty so callers never crash.
    """
    try:
        has_get_meta = hasattr(db, "get_meta")
        if has_get_meta:
            cur = db._conn.execute("SELECT MAX(timestamp_ms) FROM frames")
        else:
            cur = db.execute("SELECT MAX(timestamp_ms) FROM frames")
        row = cur.fetchone()
        if row and row[0] is not None:
            return float(row[0]) / 1000.0
    except Exception:
        pass
    return 0.0


def resolve_video(run_dir: Path, db) -> Optional[VideoSource]:
    """Locate a playable video for the given run and return a VideoSource.

    Resolution order:

    1. ``run_meta.video_path`` (via ``db.get_meta("video_path")``) if the
       file exists on disk → ``kind="original"``, identity time mapper.
    2. ``run_dir / "annotated_strip.mp4"`` (then ``.avi``) if it exists →
       ``kind="strip"``, proportional time mapper using cv2 for strip
       duration and the frames table for source duration.
    3. ``None`` when neither candidate is available.

    Never raises on a missing or moved file.

    Args:
        run_dir: Directory containing the run's ``tempograph.db``.
        db: A database handle with a ``get_meta(key)`` method (e.g. an
            instance of ``src.storage.TempoGraphDB``), or a raw
            ``sqlite3.Connection``.

    Returns:
        A :class:`VideoSource` pointing at the first usable video, or ``None``.
    """
    # 1. Try the stored original video path.
    video_path_str = None
    db_unusable = False
    try:
        if hasattr(db, "get_meta"):
            video_path_str = db.get_meta("video_path")
        else:
            cur = db.execute(
                "SELECT value FROM run_meta WHERE key = ?", ("video_path",)
            )
            row = cur.fetchone()
            video_path_str = row["value"] if row else None
    except Exception:
        # DB handle closed or otherwise unusable — mark as unusable so we
        # can fall back to scanning the run dir for any video file.
        db_unusable = True

    if video_path_str:
        original = Path(video_path_str)
        if not original.is_absolute():
            original = run_dir / original
        if original.exists() and original.is_file():
            return VideoSource(
                path=original,
                kind="original",
                time_mapper=lambda t: t,
            )

    # 1b. Fallback: if we could not read run_meta (e.g. closed db handle),
    # look for any video file in the run dir itself.
    if db_unusable:
        for candidate in sorted(run_dir.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in (
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
            ):
                # Skip annotated strips — those are handled in step 2.
                if candidate.name.startswith("annotated_strip"):
                    continue
                return VideoSource(
                    path=candidate,
                    kind="original",
                    time_mapper=lambda t: t,
                )

    # 2. Try the annotated strip.
    strip_candidates = [
        run_dir / "annotated_strip.mp4",
        run_dir / "annotated_strip.avi",
    ]
    strip_path = next((p for p in strip_candidates if p.exists()), None)
    if strip_path is not None:
        strip_dur = _probe_strip_duration(strip_path)
        source_dur = _source_duration_from_db(db)
        # If source duration is unknown (no frames / closed db), assume
        # 1:1 mapping rather than collapsing to zero.
        if source_dur <= 0:
            source_dur = strip_dur
        return VideoSource(
            path=strip_path,
            kind="strip",
            time_mapper=make_strip_mapper(source_dur, strip_dur),
        )

    return None


def _fmt_mmss(seconds: float) -> str:
    """Format seconds as ``MM:SS`` for caption text under the video player."""
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def render_player(source: VideoSource, start_s: float) -> None:
    """Render a Streamlit video player starting at *start_s* seconds.

    Emits ``st.video`` with ``start_time`` passed through
    ``source.time_mapper``, followed by a short caption stating what is
    playing (source video or annotated strip, with MM:SS start time).

    Args:
        source: A :class:`VideoSource` resolved by :func:`resolve_video`.
        start_s: The requested start time in source seconds (from the
            timestamp button that was clicked).
    """
    start_time = int(source.time_mapper(start_s))
    st.video(str(source.path), start_time=start_time)
    mmss = _fmt_mmss(source.time_mapper(start_s))
    if source.kind == "original":
        st.caption(f"source video @ {mmss}")
    else:
        st.caption(f"annotated strip (source file missing) @ {mmss}")
