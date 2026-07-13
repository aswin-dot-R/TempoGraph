"""Graph-driven clip export: turn known events into ffmpeg cuts.

``select_events`` reads a run's graph data (visual_events from
``analysis.json`` beside the DB, plus ethogram behavior labels stored in the
DB), filters by entity / behavior / time range, pads each event by ±1.5 s and
merges overlapping spans. ``export_clips`` cuts one mp4 per span
(stream-copy where keyframes allow, re-encode fallback) and can concatenate
them into a crossfaded montage with burned-in label lower-thirds.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_PAD_MS = 1500
_DURATION_TOLERANCE_S = 0.4  # tighter than the ±0.5 s acceptance bound
_MONTAGE_FADE_S = 0.25

Span = Tuple[int, int, str]  # (start_ms, end_ms, label)


# ─── span math ────────────────────────────────────────────────────────────────


def pad_and_merge(events: Sequence[Span], pad_ms: int = DEFAULT_PAD_MS) -> List[Span]:
    """Pad each (start_ms, end_ms, label) event and merge overlapping spans.

    Start times are clamped at 0. Spans that overlap or touch after padding
    are merged; merged labels join unique labels with " + " in time order.
    """
    padded = []
    for start_ms, end_ms, label in events:
        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms
        padded.append(
            (max(0, int(start_ms) - pad_ms), int(end_ms) + pad_ms, str(label))
        )
    padded.sort(key=lambda s: (s[0], s[1]))

    merged: List[List] = []
    for start_ms, end_ms, label in padded:
        if merged and start_ms <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_ms)
            if label not in merged[-1][2]:
                merged[-1][2].append(label)
        else:
            merged.append([start_ms, end_ms, [label]])
    return [(s, e, " + ".join(labels)) for s, e, labels in merged]


def _ts_to_ms(ts) -> int:
    """Parse 'MM:SS(.ff)' / 'HH:MM:SS' / numeric seconds into milliseconds."""
    if isinstance(ts, (int, float)):
        return int(float(ts) * 1000)
    parts = str(ts).split(":")
    try:
        secs = 0.0
        for p in parts:
            secs = secs * 60 + float(p)
        return int(secs * 1000)
    except ValueError:
        return 0


# ─── event selection ──────────────────────────────────────────────────────────


def _resolve_run(db) -> Tuple[Path, Path]:
    """Accept a run dir or a tempograph.db path; return (db_path, run_dir)."""
    p = Path(db)
    if p.is_dir():
        return p / "tempograph.db", p
    return p, p.parent


def _visual_events(
    run_dir: Path, entity: Optional[str], behavior: Optional[str]
) -> List[Span]:
    f = run_dir / "analysis.json"
    if not f.exists():
        return []
    try:
        analysis = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    spans: List[Span] = []
    for ev in analysis.get("visual_events", []) or []:
        ev_type = ev.get("type", "")
        if hasattr(ev_type, "value"):
            ev_type = ev_type.value
        ev_type = str(ev_type)
        entities = [str(e) for e in (ev.get("entities") or [])]
        if behavior is not None and ev_type != behavior:
            continue
        if entity is not None and entity not in entities:
            continue
        start_ms = _ts_to_ms(ev.get("start_time", 0))
        end_ms = _ts_to_ms(ev.get("end_time", ev.get("start_time", 0)))
        label = ev_type if not entities else f"{ev_type}: {', '.join(entities)}"
        spans.append((start_ms, max(start_ms, end_ms), label))
    return spans


def _ethogram_events(
    db_path: Path, entity: Optional[str], behavior: Optional[str]
) -> List[Span]:
    """Contiguous same-behavior ethogram labels -> (start_ms, end_ms, label).

    Ethogram labels are frame-level and not entity-scoped, so they only
    contribute when no entity filter is requested.
    """
    if entity is not None or not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT el.behavior, f.timestamp_ms FROM ethogram_labels el "
                "JOIN frames f ON f.frame_idx = el.frame_idx "
                "ORDER BY f.timestamp_ms ASC"
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    spans: List[Span] = []
    for row in rows:
        beh, ts = str(row["behavior"]), int(row["timestamp_ms"])
        if behavior is not None and beh != behavior:
            continue
        if spans and spans[-1][2] == beh:
            spans[-1] = (spans[-1][0], ts, beh)
        else:
            spans.append((ts, ts, beh))
    return spans


def select_events(
    db,
    entity: Optional[str] = None,
    behavior: Optional[str] = None,
    time_range: Optional[Tuple[int, int]] = None,
    pad_ms: int = DEFAULT_PAD_MS,
) -> List[Span]:
    """Select graph events for a run and return padded, merged clip spans.

    ``db``: path to a run's ``tempograph.db`` (or the run directory).
    ``entity`` / ``behavior``: exact-match filters (entity id from the graph,
    behavior/event type). ``time_range``: (start_ms, end_ms) — events must
    overlap it (applied before padding). Each surviving event is padded by
    ``pad_ms`` on both sides and overlapping spans are merged.
    """
    db_path, run_dir = _resolve_run(db)
    events = _visual_events(run_dir, entity, behavior)
    events += _ethogram_events(db_path, entity, behavior)

    if time_range is not None:
        lo, hi = int(time_range[0]), int(time_range[1])
        events = [e for e in events if e[1] >= lo and e[0] <= hi]

    return pad_and_merge(events, pad_ms=pad_ms)


# ─── ffmpeg export ────────────────────────────────────────────────────────────


def _slug(label: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return s[:max_len] or "event"


def _ffprobe_duration(path: Path) -> Optional[float]:
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
            timeout=30,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _run_ffmpeg(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error"] + args,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _cut_clip(video_path: Path, start_s: float, dur_s: float, out: Path) -> None:
    """Cut one clip: stream-copy first; re-encode when keyframes misalign."""
    copy_args = [
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{dur_s:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(out),
    ]
    proc = _run_ffmpeg(copy_args)
    if proc.returncode == 0:
        got = _ffprobe_duration(out)
        if got is not None and abs(got - dur_s) <= _DURATION_TOLERANCE_S:
            return
    # Keyframe alignment insufficient (or copy failed) — accurate re-encode.
    enc_args = [
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
    proc = _run_ffmpeg(enc_args)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed cutting {out.name}: {proc.stderr[-800:]}")


def _drawtext(label: str) -> str:
    """A lower-third drawtext filter for a (sanitised) label."""
    safe = re.sub(r"[^A-Za-z0-9 _.,+-]+", " ", label).strip()[:60] or "event"
    return (
        f"drawtext=text='{safe}':x=(w-text_w)/2:y=h-(2*text_h):"
        f"fontsize=h/18:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10"
    )


def _build_montage(
    clips: List[Path], labels: List[str], out: Path, fade_s: float = _MONTAGE_FADE_S
) -> None:
    durations = []
    for c in clips:
        d = _ffprobe_duration(c)
        if d is None:
            raise RuntimeError(f"ffprobe failed on {c}")
        durations.append(d)

    def _filtergraph(with_labels: bool) -> str:
        parts = []
        for i, label in enumerate(labels):
            chain = "fps=30,scale=640:-2,setsar=1,format=yuv420p"
            if with_labels:
                chain += "," + _drawtext(label)
            parts.append(f"[{i}:v]{chain}[v{i}]")
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
        return ";".join(parts)

    inputs: List[str] = []
    for c in clips:
        inputs += ["-i", str(c)]

    for with_labels in (True, False):
        args = inputs + [
            "-filter_complex",
            _filtergraph(with_labels),
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
        if proc.returncode == 0:
            return
    raise RuntimeError(f"ffmpeg montage failed: {proc.stderr[-800:]}")


def export_clips(
    video_path,
    spans: Sequence[Span],
    out_dir,
    montage: bool = False,
) -> Dict:
    """Cut one mp4 per (start_ms, end_ms, label) span from ``video_path``.

    Returns ``{"clips": [Path, ...], "montage": Path | None}``. Spans are
    clamped to the source duration; ``montage=True`` additionally writes
    ``montage.mp4`` — the clips concatenated with crossfades and burned-in
    label lower-thirds.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        raise FileNotFoundError(f"source video not found: {video_path}")

    source_dur = _ffprobe_duration(video_path)
    clips: List[Path] = []
    labels: List[str] = []
    for i, (start_ms, end_ms, label) in enumerate(spans):
        start_s = max(0.0, start_ms / 1000.0)
        end_s = end_ms / 1000.0
        if source_dur is not None:
            start_s = min(start_s, max(0.0, source_dur - 0.1))
            end_s = min(end_s, source_dur)
        dur_s = end_s - start_s
        if dur_s <= 0:
            continue
        out = out_dir / f"clip_{i:03d}_{_slug(label)}.mp4"
        _cut_clip(video_path, start_s, dur_s, out)
        clips.append(out)
        labels.append(label)

    montage_path: Optional[Path] = None
    if montage and clips:
        montage_path = out_dir / "montage.mp4"
        _build_montage(clips, labels, montage_path)

    return {"clips": clips, "montage": montage_path}
