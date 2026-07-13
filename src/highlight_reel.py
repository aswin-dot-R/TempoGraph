"""Highlight reel: pick top spans by delta_score, cut with ffmpeg.

``pick_highlight_spans`` is pure span math — it reads the ``frames`` table,
sorts by ``delta_score`` descending, and greedily accepts spans separated by
at least ``min_gap_s``. ``build_highlight_reel`` handles the ffmpeg work:
cutting each span with keyframe-accurate re-encode, concatenating them with
xfade crossfades, ffprobe-verification, and returning the output path.

Mirrors the structure of ``src/clip_export.py``: span math separated from
ffmpeg work, reuse of the same ``ffprobe`` / ``ffmpeg`` helpers.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Tuple

_DURATION_TOLERANCE_S = 0.4
_MONTAGE_FADE_S = 0.25
_FFPROBE_TIMEOUT = 30
_FFMPEG_TIMEOUT = 600


# ─── span math (pure) ─────────────────────────────────────────────────────────


def _resolve_db(p: Path) -> Path:
    """Accept a run dir or a ``tempograph.db`` path; return the DB path.

    Mirrors ``src/clip_export._resolve_run``.
    """
    if p.is_dir():
        return p / "tempograph.db"
    return p


def _db_frames(db_path: Path) -> List[Tuple[int, int]]:
    """Return [(timestamp_ms, delta_score)] for every row, sorted by score desc.

    Returns an empty list if the DB is missing or the frames table doesn't
    exist.
    """
    db_path = _resolve_db(db_path)
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT timestamp_ms, delta_score FROM frames ORDER BY delta_score DESC"
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(int(ts), float(score)) for ts, score in rows]


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
    pad_ms = int(span_padding_s * 1000)
    min_gap_ms = int(min_gap_s * 1000)
    target_ms = int(target_duration_s * 1000)

    frames = _db_frames(db_path)

    accepted: List[Tuple[int, int]] = []  # list of (start_ms, end_ms) padded spans

    for ts_ms, _score in frames:
        start = max(0, ts_ms - pad_ms)
        end = ts_ms + pad_ms

        # Check minimum gap against every accepted span.
        gap_ok = True
        for as_, ae in accepted:
            # Overlapping or touching padded spans are rejected.
            if start <= ae and as_ <= end:
                gap_ok = False
                break
            # Non-overlapping: compute the gap between them.
            if end <= as_:
                # Current span is entirely before accepted span.
                space = as_ - end
            else:
                # Accepted span is entirely before current span.
                space = start - ae
            if space < min_gap_ms:
                gap_ok = False
                break
        if not gap_ok:
            continue

        accepted.append((start, end))

        # Sum the current total duration and stop if we've reached the target.
        total = sum(e - s for s, e in accepted)
        if total >= target_ms:
            break

    # Merge overlapping / touching spans.
    if not accepted:
        return []
    accepted.sort()
    merged: List[List[int]] = [[accepted[0][0], accepted[0][1]]]
    for s, e in accepted[1:]:
        if s <= merged[-1][1]:  # overlapping or touching
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(int(ms), int(me)) for ms, me in merged]


# ─── ffmpeg helpers (mirror src/clip_export.py) ───────────────────────────────


def _ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _run_ffmpeg(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error"] + args,
        capture_output=True,
        text=True,
        timeout=_FFMPEG_TIMEOUT,
    )


def _cut_clip(video_path: Path, start_s: float, dur_s: float, out: Path) -> None:
    """Cut one clip via accurate re-encode (frame-accurate)."""
    args = [
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{dur_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out),
    ]
    proc = _run_ffmpeg(args)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed cutting clip {out.name}: {proc.stderr[-800:]}"
        )


def _slug(label: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return s[:max_len] or "clip"


def _build_concat_fade(
    clips: List[Path],
    out: Path,
    fade_s: float,
) -> None:
    """Concatenate clips with xfade crossfades between them."""
    durations = []
    for c in clips:
        d = _ffprobe_duration(c)
        if d is None:
            raise RuntimeError(f"ffprobe failed on {c}")
        durations.append(d)

    # Build filtergraph: each clip → scale to yuv420p, then chain xfade.
    parts: List[str] = []
    for i in range(len(clips)):
        parts.append(f"[{i}:v]fps=30,scale=640:-2,setsar=1,format=yuv420p[v{i}]")
    if len(clips) == 1:
        parts.append("[v0]null[vout]")
    else:
        prev = "v0"
        offset = 0.0
        for i in range(1, len(clips)):
            offset += durations[i - 1] - fade_s
            nxt = f"x{i}" if i < len(clips) - 1 else "vout"
            parts.append(
                f"[{prev}][v{i}]xfade=transition=fade:"
                f"duration={fade_s}:offset={offset:.3f}[{nxt}]"
            )
            prev = nxt

    inputs: List[str] = []
    for c in clips:
        inputs += ["-i", str(c)]

    args = inputs + [
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[vout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out),
    ]
    proc = _run_ffmpeg(args)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat/fade failed: {proc.stderr[-800:]}")


# ─── public API ───────────────────────────────────────────────────────────────


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
    if not spans:
        raise ValueError("build_highlight_reel: spans list is empty")

    video_path = Path(video_path)
    if not video_path.exists():
        raise RuntimeError(f"source video not found: {video_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_dur = _ffprobe_duration(video_path)

    # Cut each span into a temporary clip.
    tmp_clips: List[Path] = []
    for i, (start_ms, end_ms) in enumerate(spans):
        start_s = max(0.0, start_ms / 1000.0)
        end_s = min(end_ms / 1000.0, source_dur) if source_dur else end_ms / 1000.0
        dur_s = end_s - start_s
        if dur_s <= 0:
            continue
        tmp_out = out_path.with_name(
            f"_hl_clip_{i:03d}_{_slug(f'{start_ms}-{end_ms}')}.mp4"
        )
        _cut_clip(video_path, start_s, dur_s, tmp_out)
        tmp_clips.append(tmp_out)

    if not tmp_clips:
        raise RuntimeError(f"ffmpeg produced no valid clips from {len(spans)} span(s)")

    try:
        if len(tmp_clips) == 1:
            # Single span: just verify the cut, then rename.
            dur = _ffprobe_duration(tmp_clips[0])
            want = (spans[0][1] - spans[0][0]) / 1000.0
            if dur is None or abs(dur - want) > _DURATION_TOLERANCE_S:
                raise RuntimeError(
                    f"ffprobe duration mismatch: got {dur}, want ~{want}"
                )
            tmp_clips[0].rename(out_path)
        else:
            _build_concat_fade(tmp_clips, out_path, fade_s)
    finally:
        # Clean up temporary clips.
        for c in tmp_clips:
            try:
                c.unlink()
            except OSError:
                pass

    # Verify final output.
    if not out_path.exists():
        raise RuntimeError(f"highlight reel output not found: {out_path}")
    return out_path
