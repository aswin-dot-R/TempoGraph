# TempoGraph v2 — Chunked VLM Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic "all frames in one VLM call" pipeline with a chunked, motion-aware pipeline that uses llama-server exclusively, persists per-frame YOLO/depth in SQLite, captions video in 10-frame chunks with hard-wipe context, and aggregates captions into the existing `AnalysisResult` shape.

**Architecture:** Six sequential stages — frame selection (delta- or motion-compensated keyframes), YOLO sweep into SQLite, optional depth, top-K frame scoring for VLM, chunked VLM with one-line summary seeds, and a final aggregation LLM call producing `AnalysisResult`. Existing `GraphBuilder`, `JSONParser`, `models.py` are reused unchanged.

**Tech Stack:** Python 3.10+, OpenCV (ORB + RANSAC homography), Ultralytics YOLO11, Depth Anything V2, SQLite (stdlib), Ollama HTTP via `requests`, Streamlit + Plotly, pytest.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/models.py` | modify | Add `CameraMode` enum, `FrameSelectionResult`, `ChunkCaption` dataclasses |
| `src/storage.py` | new | SQLite schema + CRUD for frames, detections, depth, captions |
| `src/modules/frame_selector.py` | new | Two-mode frame selection (static delta / motion-compensated), produces `FrameSelectionResult` |
| `src/modules/detector.py` | modify | Add `detect_to_db(frame_paths, db)` method storing rows in `detections` table |
| `src/modules/depth.py` | modify | Add `estimate_to_db(frame_paths, db, output_dir)` storing per-bbox mean depth |
| `src/modules/frame_scorer.py` | new | Score YOLO frames, return top-K indices for VLM |
| `src/backends/llama_server_backend.py` | modify | Add `caption_chunks(chunks, db) -> List[ChunkCaption]` with hard-wipe seed |
| `src/aggregator.py` | new | Captions → `AnalysisResult` via second llama-server call (single + hierarchical) |
| `src/pipeline_v2.py` | new | New orchestrator wiring stages 1-6 with SQLite-backed flow |
| `ui/app.py` | modify | Camera mode radio, separate FPS sliders, "Preview frame selection" button + delta plot |
| `tests/test_storage.py` | new | SQLite CRUD round-trips |
| `tests/test_frame_selector.py` | new | Static + motion-compensated keyframe selection on synthetic video |
| `tests/test_frame_scorer.py` | new | Top-K selection prefers detection-set changes over pure delta |
| `tests/test_chunked_vlm.py` | new | Mocked Ollama: chunking, seed propagation, parsing |
| `tests/test_aggregator.py` | new | Mocked Ollama: single-pass + hierarchical aggregation |

`graph_builder.py`, `video_annotator.py`, `json_parser.py`, `api.py` are not modified.

---

## Task 1: Add v2 data models

**Files:**
- Modify: `src/models.py`
- Test: `tests/test_models_v2.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_v2.py`:

```python
from src.models import CameraMode, FrameSelectionResult, ChunkCaption


def test_camera_mode_enum_values():
    assert CameraMode.STATIC.value == "static"
    assert CameraMode.MOVING.value == "moving"
    assert CameraMode.AUTO.value == "auto"


def test_frame_selection_result_fields():
    r = FrameSelectionResult(
        frame_indices=[0, 5, 10],
        keyframe_indices=[0, 10],
        sampled_indices=[5],
        scan_indices=[0, 5, 10, 15],
        deltas=[0.0, 1.5, 12.3, 0.4],
        threshold=2.0,
        camera_mode=CameraMode.STATIC,
    )
    assert r.frame_indices == [0, 5, 10]
    assert r.keyframe_indices == [0, 10]
    assert r.threshold == 2.0


def test_chunk_caption_fields():
    c = ChunkCaption(
        chunk_id=0,
        frame_indices=[10, 20, 30],
        per_frame_lines={10: "person enters", 20: "(no change)", 30: "person sits"},
        summary="A person entered and sat down.",
        raw_response="FRAME_10: ...",
    )
    assert c.chunk_id == 0
    assert c.summary.startswith("A person")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models_v2.py -v
```
Expected: ImportError — `CameraMode`, `FrameSelectionResult`, `ChunkCaption` not defined.

- [ ] **Step 3: Add models to `src/models.py`**

Append to `src/models.py`:

```python
from dataclasses import dataclass, field


class CameraMode(str, Enum):
    STATIC = "static"
    MOVING = "moving"
    AUTO = "auto"


@dataclass
class FrameSelectionResult:
    frame_indices: List[int]        # union of keyframe + sampled, sorted
    keyframe_indices: List[int]     # green: above-threshold delta frames
    sampled_indices: List[int]      # orange: uniform-FPS samples
    scan_indices: List[int]         # frame index for each delta below
    deltas: List[float]             # parallel to scan_indices
    threshold: float                # delta threshold used
    camera_mode: CameraMode


@dataclass
class ChunkCaption:
    chunk_id: int
    frame_indices: List[int]
    per_frame_lines: dict           # {frame_idx: caption_str}
    summary: str
    raw_response: str
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_models_v2.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models_v2.py
git commit -m "v2: add CameraMode, FrameSelectionResult, ChunkCaption models"
```

---

## Task 2: SQLite storage layer

**Files:**
- Create: `src/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_storage.py`:

```python
import tempfile
from pathlib import Path

from src.storage import TempoGraphDB


def test_db_creates_schema():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        assert db.has_table("frames")
        assert db.has_table("detections")
        assert db.has_table("depth_frames")
        db.close()


def test_insert_and_get_frame():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        db.insert_frame(
            frame_idx=10,
            timestamp_ms=333,
            image_path="/tmp/f.jpg",
            is_keyframe=True,
            delta_score=8.4,
        )
        row = db.get_frame(10)
        assert row["frame_idx"] == 10
        assert row["timestamp_ms"] == 333
        assert row["is_keyframe"] == 1
        assert row["delta_score"] == 8.4
        db.close()


def test_insert_detections_and_query_by_frame():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        db.insert_frame(frame_idx=5, timestamp_ms=166, image_path="/tmp/5.jpg", is_keyframe=False, delta_score=0.1)
        db.insert_detection(
            frame_idx=5, track_id=1, class_name="person",
            x1=0.1, y1=0.2, x2=0.4, y2=0.9, confidence=0.92,
            mean_depth=None,
        )
        db.insert_detection(
            frame_idx=5, track_id=2, class_name="dog",
            x1=0.5, y1=0.6, x2=0.7, y2=0.85, confidence=0.81,
            mean_depth=None,
        )
        rows = db.get_detections_for_frame(5)
        assert len(rows) == 2
        classes = {r["class_name"] for r in rows}
        assert classes == {"person", "dog"}
        db.close()


def test_set_mean_depth_updates_detection():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        db.insert_frame(frame_idx=1, timestamp_ms=0, image_path="/tmp/1.jpg", is_keyframe=False, delta_score=0.0)
        det_id = db.insert_detection(
            frame_idx=1, track_id=1, class_name="person",
            x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=0.9,
            mean_depth=None,
        )
        db.set_detection_mean_depth(det_id, 0.42)
        rows = db.get_detections_for_frame(1)
        assert rows[0]["mean_depth"] == 0.42
        db.close()


def test_get_all_frame_indices_sorted():
    with tempfile.TemporaryDirectory() as td:
        db = TempoGraphDB(Path(td) / "t.db")
        for idx in [10, 2, 7, 5]:
            db.insert_frame(frame_idx=idx, timestamp_ms=idx * 33, image_path=f"/tmp/{idx}.jpg", is_keyframe=False, delta_score=0.0)
        assert db.get_all_frame_indices() == [2, 5, 7, 10]
        db.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_storage.py -v
```
Expected: ImportError — `src.storage` does not exist.

- [ ] **Step 3: Implement `src/storage.py`**

```python
"""SQLite-backed storage for TempoGraph v2 pipeline."""

import sqlite3
from pathlib import Path
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_idx INTEGER PRIMARY KEY,
    timestamp_ms INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    is_keyframe INTEGER NOT NULL,
    delta_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_idx INTEGER NOT NULL,
    track_id INTEGER,
    class_name TEXT NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    confidence REAL NOT NULL,
    mean_depth REAL,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);

CREATE TABLE IF NOT EXISTS depth_frames (
    frame_idx INTEGER PRIMARY KEY,
    depth_npy_path TEXT NOT NULL,
    FOREIGN KEY (frame_idx) REFERENCES frames(frame_idx)
);

CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_idx);
"""


class TempoGraphDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def has_table(self, name: str) -> bool:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        return cur.fetchone() is not None

    def insert_frame(
        self,
        frame_idx: int,
        timestamp_ms: int,
        image_path: str,
        is_keyframe: bool,
        delta_score: float,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO frames (frame_idx, timestamp_ms, image_path, is_keyframe, delta_score) "
            "VALUES (?, ?, ?, ?, ?)",
            (frame_idx, timestamp_ms, image_path, 1 if is_keyframe else 0, delta_score),
        )
        self._conn.commit()

    def get_frame(self, frame_idx: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM frames WHERE frame_idx = ?", (frame_idx,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_frame_indices(self) -> list:
        rows = self._conn.execute(
            "SELECT frame_idx FROM frames ORDER BY frame_idx ASC"
        ).fetchall()
        return [r["frame_idx"] for r in rows]

    def insert_detection(
        self,
        frame_idx: int,
        track_id: Optional[int],
        class_name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        mean_depth: Optional[float] = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO detections "
            "(frame_idx, track_id, class_name, x1, y1, x2, y2, confidence, mean_depth) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (frame_idx, track_id, class_name, x1, y1, x2, y2, confidence, mean_depth),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_detections_for_frame(self, frame_idx: int) -> list:
        rows = self._conn.execute(
            "SELECT * FROM detections WHERE frame_idx = ? ORDER BY detection_id ASC",
            (frame_idx,),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_detection_mean_depth(self, detection_id: int, mean_depth: float) -> None:
        self._conn.execute(
            "UPDATE detections SET mean_depth = ? WHERE detection_id = ?",
            (mean_depth, detection_id),
        )
        self._conn.commit()

    def insert_depth_frame(self, frame_idx: int, depth_npy_path: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO depth_frames (frame_idx, depth_npy_path) VALUES (?, ?)",
            (frame_idx, depth_npy_path),
        )
        self._conn.commit()

    def get_depth_path(self, frame_idx: int) -> Optional[str]:
        row = self._conn.execute(
            "SELECT depth_npy_path FROM depth_frames WHERE frame_idx = ?",
            (frame_idx,),
        ).fetchone()
        return row["depth_npy_path"] if row else None

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_storage.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/storage.py tests/test_storage.py
git commit -m "v2: SQLite storage layer for frames, detections, depth"
```

---

## Task 3: Frame selector — static camera mode

**Files:**
- Create: `src/modules/frame_selector.py`
- Test: `tests/test_frame_selector.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frame_selector.py`:

```python
import numpy as np
import cv2
import tempfile
from pathlib import Path

from src.modules.frame_selector import FrameSelector
from src.models import CameraMode


def _make_synthetic_video(path: Path, n_frames: int = 60, w: int = 320, h: int = 240) -> None:
    """Create a synthetic video where frames 20-25 have a sudden brightness shift."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for i in range(n_frames):
        if 20 <= i < 25:
            frame = np.full((h, w, 3), 200, dtype=np.uint8)
        else:
            frame = np.full((h, w, 3), 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_static_mode_finds_keyframes_at_brightness_shift():
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "v.mp4"
        _make_synthetic_video(video)

        selector = FrameSelector()
        result = selector.select(
            video_path=str(video),
            camera_mode=CameraMode.STATIC,
            sample_fps=1.0,
            threshold_mult=1.0,
        )

        assert result.camera_mode == CameraMode.STATIC
        assert len(result.frame_indices) > 0
        # The frame at the brightness boundary should be a keyframe
        assert any(20 <= kf <= 25 for kf in result.keyframe_indices)


def test_static_mode_uniform_samples_at_user_fps():
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "v.mp4"
        _make_synthetic_video(video, n_frames=120)  # 4 seconds at 30fps

        selector = FrameSelector()
        result = selector.select(
            video_path=str(video),
            camera_mode=CameraMode.STATIC,
            sample_fps=1.0,
            threshold_mult=1.0,
        )

        # 1 Hz over 4 seconds -> roughly 4 sampled frames (give or take 1)
        assert 3 <= len(result.sampled_indices) <= 5


def test_returns_deltas_and_threshold_for_plotting():
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "v.mp4"
        _make_synthetic_video(video)

        result = FrameSelector().select(
            video_path=str(video), camera_mode=CameraMode.STATIC, sample_fps=1.0
        )

        assert len(result.deltas) == len(result.scan_indices)
        assert result.threshold > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_frame_selector.py -v
```
Expected: ImportError — `src.modules.frame_selector` not found.

- [ ] **Step 3: Implement `src/modules/frame_selector.py`**

```python
"""Two-mode frame selection for TempoGraph v2.

Static mode: pixel-delta keyframes + uniform sampling.
Moving mode: motion-compensated residual delta after homography warp.
Auto: detect mode from first 30 sampled frames' ORB displacement.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from src.models import CameraMode, FrameSelectionResult


class FrameSelector:
    def __init__(self, thumb_width: int = 160, orb_features: int = 500):
        self.thumb_width = thumb_width
        self.orb_features = orb_features
        self.logger = logging.getLogger(__name__)

    def select(
        self,
        video_path: str,
        camera_mode: CameraMode = CameraMode.STATIC,
        sample_fps: float = 1.0,
        threshold_mult: float = 1.0,
    ) -> FrameSelectionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scan_interval = max(1, int(video_fps / max(sample_fps * 2, 2.0)))

        if camera_mode == CameraMode.AUTO:
            camera_mode = self._auto_detect_mode(cap, scan_interval, width, height)

        if camera_mode == CameraMode.STATIC:
            scan_indices, deltas = self._scan_pixel_deltas(
                cap, scan_interval, width, height
            )
        else:
            scan_indices, deltas = self._scan_motion_compensated_deltas(
                cap, scan_interval, width, height
            )

        threshold = self._compute_threshold(deltas, threshold_mult)
        keyframe_indices = self._extract_keyframes(scan_indices, deltas, threshold)
        sampled_indices = self._uniform_sample(total, video_fps, sample_fps)

        union = sorted(set(keyframe_indices) | set(sampled_indices))

        cap.release()

        self.logger.info(
            f"Frame selection ({camera_mode.value}): {len(union)} frames "
            f"({len(keyframe_indices)} keyframe + {len(sampled_indices)} sampled), "
            f"threshold={threshold:.2f}"
        )

        return FrameSelectionResult(
            frame_indices=union,
            keyframe_indices=keyframe_indices,
            sampled_indices=sampled_indices,
            scan_indices=scan_indices,
            deltas=deltas,
            threshold=threshold,
            camera_mode=camera_mode,
        )

    def _scan_pixel_deltas(
        self, cap: cv2.VideoCapture, scan_interval: int, width: int, height: int
    ) -> Tuple[List[int], List[float]]:
        thumb_w = min(self.thumb_width, width)
        thumb_h = max(1, int(height * thumb_w / width)) if width > 0 else 120

        indices: List[int] = []
        deltas: List[float] = []
        prev = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % scan_interval == 0:
                small = cv2.resize(frame, (thumb_w, thumb_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev is None:
                    deltas.append(0.0)
                else:
                    deltas.append(float(np.mean(np.abs(gray - prev))))
                indices.append(i)
                prev = gray
            i += 1
        return indices, deltas

    def _scan_motion_compensated_deltas(
        self, cap: cv2.VideoCapture, scan_interval: int, width: int, height: int
    ) -> Tuple[List[int], List[float]]:
        thumb_w = min(self.thumb_width, width)
        thumb_h = max(1, int(height * thumb_w / width)) if width > 0 else 120
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        indices: List[int] = []
        deltas: List[float] = []
        prev_small = None
        prev_gray = None
        prev_kp = None
        prev_des = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % scan_interval == 0:
                small = cv2.resize(frame, (thumb_w, thumb_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)

                if prev_gray is None or des is None or prev_des is None:
                    deltas.append(0.0)
                else:
                    delta = self._residual_after_warp(
                        prev_gray, gray, prev_kp, kp, prev_des, des, bf
                    )
                    deltas.append(delta)
                indices.append(i)
                prev_gray = gray
                prev_kp = kp
                prev_des = des
            i += 1
        return indices, deltas

    def _residual_after_warp(
        self, prev_gray, gray, prev_kp, kp, prev_des, des, bf
    ) -> float:
        try:
            matches = bf.match(prev_des, des)
            if len(matches) < 8:
                return float(np.mean(np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32))))

            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if H is None:
                return float(np.mean(np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32))))

            h, w = gray.shape
            warped = cv2.warpPerspective(prev_gray, H, (w, h))
            return float(np.mean(np.abs(gray.astype(np.float32) - warped.astype(np.float32))))
        except cv2.error:
            return float(np.mean(np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32))))

    def _compute_threshold(self, deltas: List[float], mult: float) -> float:
        non_zero = [d for d in deltas if d > 0]
        if not non_zero:
            return 0.0
        arr = np.array(non_zero)
        return float(np.median(arr) + mult * np.std(arr))

    def _extract_keyframes(
        self, indices: List[int], deltas: List[float], threshold: float
    ) -> List[int]:
        kf = [indices[i] for i, d in enumerate(deltas) if d >= threshold and threshold > 0]
        # Always include first frame
        if indices and indices[0] not in kf:
            kf.insert(0, indices[0])
        return sorted(set(kf))

    def _uniform_sample(self, total_frames: int, video_fps: float, sample_fps: float) -> List[int]:
        if sample_fps <= 0 or video_fps <= 0:
            return []
        step = max(1, int(round(video_fps / sample_fps)))
        return list(range(0, total_frames, step))

    def _auto_detect_mode(
        self, cap: cv2.VideoCapture, scan_interval: int, width: int, height: int
    ) -> CameraMode:
        """Estimate dominant ORB displacement over first 30 sampled frames."""
        thumb_w = min(self.thumb_width, width)
        thumb_h = max(1, int(height * thumb_w / width))
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_kp = None
        prev_des = None
        displacements: List[float] = []
        i = 0
        sampled = 0
        while sampled < 30:
            ret, frame = cap.read()
            if not ret:
                break
            if i % scan_interval == 0:
                small = cv2.resize(frame, (thumb_w, thumb_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)
                if prev_des is not None and des is not None and len(kp) > 0 and len(prev_kp) > 0:
                    matches = bf.match(prev_des, des)
                    if matches:
                        d = np.mean(
                            [np.linalg.norm(np.array(prev_kp[m.queryIdx].pt) - np.array(kp[m.trainIdx].pt)) for m in matches]
                        )
                        displacements.append(d)
                prev_kp = kp
                prev_des = des
                sampled += 1
            i += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind for caller

        if not displacements:
            return CameraMode.STATIC
        median_disp = float(np.median(displacements))
        threshold_disp = 0.05 * thumb_w
        return CameraMode.MOVING if median_disp > threshold_disp else CameraMode.STATIC
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_frame_selector.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/modules/frame_selector.py tests/test_frame_selector.py
git commit -m "v2: frame selector with static + motion-compensated modes"
```

---

## Task 4: Frame selector — moving camera test

**Files:**
- Modify: `tests/test_frame_selector.py`

- [ ] **Step 1: Add a moving-camera test that verifies homography compensation**

Append to `tests/test_frame_selector.py`:

```python
def test_moving_mode_compensates_pure_pan():
    """A pure pan should yield low residual delta (motion is cancelled)."""
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "v.mp4"
        # Synthetic pan: a textured frame translated horizontally over time
        w, h = 320, 240
        rng = np.random.default_rng(42)
        base = rng.integers(0, 255, (h, w * 2, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video), fourcc, 30.0, (w, h))
        for i in range(60):
            shift = i * 2  # 2 pixels per frame
            frame = base[:, shift:shift + w].copy()
            writer.write(frame)
        writer.release()

        result = FrameSelector().select(
            video_path=str(video),
            camera_mode=CameraMode.MOVING,
            sample_fps=1.0,
            threshold_mult=2.0,
        )

        # With motion compensation, residual deltas should be small overall.
        # We expect very few keyframes (no real scene change occurred).
        assert len(result.keyframe_indices) <= 3
```

- [ ] **Step 2: Run test to verify it passes**

```bash
pytest tests/test_frame_selector.py::test_moving_mode_compensates_pure_pan -v
```
Expected: PASS — homography cancels the pan.

If it fails: add small debug logging in `_residual_after_warp` and confirm matches > 8 (raise `orb_features` to 1000 if needed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_frame_selector.py
git commit -m "v2: test moving-camera homography compensation"
```

---

## Task 5: YOLO detect_to_db method

**Files:**
- Modify: `src/modules/detector.py`
- Test: `tests/test_detector_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detector_db.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.storage import TempoGraphDB
from src.modules.detector import ObjectDetector


def _make_jpg(path: Path) -> None:
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_detect_to_db_inserts_rows(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        f1 = td / "f1.jpg"
        f2 = td / "f2.jpg"
        _make_jpg(f1)
        _make_jpg(f2)

        db_path = td / "t.db"
        db = TempoGraphDB(db_path)
        db.insert_frame(0, 0, str(f1), True, 0.0)
        db.insert_frame(1, 1000, str(f2), False, 0.0)

        # Mock YOLO model: return one box per frame
        fake_box = MagicMock()
        fake_box.xyxy = [np.array([10.0, 20.0, 100.0, 200.0])]
        fake_box.conf = [0.85]
        fake_box.cls = [0]

        fake_result = MagicMock()
        fake_result.boxes = [fake_box]
        fake_result.boxes.id = None
        fake_result.boxes.__len__ = lambda s: 1

        fake_model = MagicMock()
        fake_model.names = {0: "person"}
        fake_model.track = MagicMock(return_value=[fake_result])

        det = ObjectDetector(device="cpu")
        det._model = fake_model

        det.detect_to_db(
            frame_indices=[0, 1],
            frame_paths=[f1, f2],
            db=db,
            frame_width=320,
            frame_height=240,
        )

        rows0 = db.get_detections_for_frame(0)
        rows1 = db.get_detections_for_frame(1)
        assert len(rows0) == 1
        assert len(rows1) == 1
        # Coords should be normalized into [0,1]
        assert 0.0 <= rows0[0]["x1"] <= 1.0
        assert 0.0 <= rows0[0]["y2"] <= 1.0
        assert rows0[0]["class_name"] == "person"
        db.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_detector_db.py -v
```
Expected: AttributeError — `ObjectDetector` has no `detect_to_db`.

- [ ] **Step 3: Add the method to `src/modules/detector.py`**

Append a new method below `detect_frames`:

```python
    def detect_to_db(
        self,
        frame_indices,
        frame_paths,
        db,
        frame_width: int,
        frame_height: int,
    ):
        """Run detection on frames and insert normalized rows into the DB."""
        self._load_model()
        self.logger.info(f"Detect-to-DB: {len(frame_paths)} frames")

        for frame_idx, frame_path in zip(frame_indices, frame_paths):
            try:
                frame = self._read_frame(frame_path)
                results = self._model.track(
                    frame,
                    conf=self.confidence,
                    imgsz=self.imgsz,
                    persist=True,
                    verbose=False,
                )
                if not results:
                    continue
                result = results[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                track_ids = None
                if result.boxes.id is not None:
                    track_ids = result.boxes.id

                for i, box in enumerate(result.boxes):
                    xyxy = box.xyxy[0]
                    if hasattr(xyxy, "cpu"):
                        xyxy = xyxy.cpu().numpy()
                    x1, y1, x2, y2 = [float(v) for v in xyxy]

                    conf = box.conf[0] if hasattr(box.conf, "__getitem__") else box.conf
                    conf = float(conf)
                    cls_idx = int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)
                    class_name = self._model.names[cls_idx]

                    track_id = None
                    if track_ids is not None and i < len(track_ids):
                        try:
                            track_id = int(track_ids[i])
                        except (TypeError, ValueError):
                            track_id = None

                    db.insert_detection(
                        frame_idx=int(frame_idx),
                        track_id=track_id,
                        class_name=class_name,
                        x1=x1 / frame_width,
                        y1=y1 / frame_height,
                        x2=x2 / frame_width,
                        y2=y2 / frame_height,
                        confidence=conf,
                    )
            except Exception as e:
                self.logger.warning(f"detect_to_db error frame {frame_idx}: {e}")
                continue
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_detector_db.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/modules/detector.py tests/test_detector_db.py
git commit -m "v2: ObjectDetector.detect_to_db writes normalized boxes to SQLite"
```

---

## Task 6: Depth → DB with per-bbox mean depth

**Files:**
- Modify: `src/modules/depth.py`
- Test: `tests/test_depth_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_depth_db.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from src.storage import TempoGraphDB
from src.modules.depth import DepthEstimator


def _make_jpg(path: Path) -> None:
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_estimate_to_db_writes_depth_and_per_bbox_means(tmp_path):
    f1 = tmp_path / "f1.jpg"
    _make_jpg(f1)

    db = TempoGraphDB(tmp_path / "t.db")
    db.insert_frame(0, 0, str(f1), True, 0.0)
    det_id = db.insert_detection(
        frame_idx=0, track_id=1, class_name="person",
        x1=0.0, y1=0.0, x2=0.5, y2=0.5, confidence=0.9,
    )

    # Mock depth model: returns a depth map where left-half=0.2, right-half=0.8
    fake_depth = np.zeros((240, 320), dtype=np.float32)
    fake_depth[:, :160] = 0.2
    fake_depth[:, 160:] = 0.8

    fake_model = MagicMock()
    fake_model.infer_image = MagicMock(return_value=fake_depth)

    estimator = DepthEstimator(device="cpu")
    estimator._model = fake_model

    estimator.estimate_to_db(
        frame_indices=[0],
        frame_paths=[f1],
        db=db,
        output_dir=str(tmp_path / "depth"),
    )

    # Depth file written
    assert db.get_depth_path(0) is not None
    # Per-bbox mean depth populated; bbox covers top-left quadrant (left half) -> ~0.2
    rows = db.get_detections_for_frame(0)
    assert rows[0]["mean_depth"] is not None
    assert abs(rows[0]["mean_depth"] - 0.2) < 0.05
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_depth_db.py -v
```
Expected: AttributeError — no `estimate_to_db` method.

- [ ] **Step 3: Add `estimate_to_db` to `src/modules/depth.py`**

Append below `estimate_frames`:

```python
    def estimate_to_db(
        self,
        frame_indices,
        frame_paths,
        db,
        output_dir: str,
    ) -> None:
        """Run depth estimation; save .npy per frame; populate per-bbox mean depth."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        self._load_model()
        self.logger.info(f"Depth-to-DB: {len(frame_paths)} frames")

        for frame_idx, frame_path in zip(frame_indices, frame_paths):
            try:
                raw = cv2.imread(str(frame_path))
                if raw is None:
                    self.logger.warning(f"Could not read {frame_path}")
                    continue
                depth = self._model.infer_image(raw)

                # Normalize to [0,1] for stable per-bbox averaging
                d_min, d_max = float(depth.min()), float(depth.max())
                if d_max > d_min:
                    depth_norm = (depth - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(depth)

                npy_path = os.path.join(output_dir, f"depth_{frame_idx:06d}.npy")
                np.save(npy_path, depth_norm.astype(np.float32))
                db.insert_depth_frame(frame_idx=int(frame_idx), depth_npy_path=npy_path)

                # Per-bbox mean depth
                h, w = depth_norm.shape[:2]
                for det in db.get_detections_for_frame(int(frame_idx)):
                    bx1 = max(0, int(det["x1"] * w))
                    by1 = max(0, int(det["y1"] * h))
                    bx2 = min(w, int(det["x2"] * w))
                    by2 = min(h, int(det["y2"] * h))
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    region = depth_norm[by1:by2, bx1:bx2]
                    if region.size == 0:
                        continue
                    mean_d = float(np.mean(region))
                    db.set_detection_mean_depth(det["detection_id"], mean_d)
            except Exception as e:
                self.logger.warning(f"depth-to-db error frame {frame_idx}: {e}")
                continue
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_depth_db.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/modules/depth.py tests/test_depth_db.py
git commit -m "v2: DepthEstimator.estimate_to_db with per-bbox mean depth"
```

---

## Task 7: Frame scorer — top-K VLM frame selection

**Files:**
- Create: `src/modules/frame_scorer.py`
- Test: `tests/test_frame_scorer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frame_scorer.py`:

```python
import tempfile
from pathlib import Path

from src.storage import TempoGraphDB
from src.modules.frame_scorer import FrameScorer


def test_scorer_prefers_detection_set_changes(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    # 3 frames; frame 1 has just a person, frame 2 has person+dog (new entity), frame 3 unchanged
    for idx in [0, 1, 2]:
        db.insert_frame(idx, idx * 1000, str(tmp_path / f"{idx}.jpg"), False, 0.5)

    db.insert_detection(0, 1, "person", 0.1, 0.1, 0.4, 0.9, 0.9)
    db.insert_detection(1, 1, "person", 0.1, 0.1, 0.4, 0.9, 0.9)
    db.insert_detection(1, 2, "dog", 0.5, 0.5, 0.7, 0.9, 0.85)
    db.insert_detection(2, 1, "person", 0.1, 0.1, 0.4, 0.9, 0.9)
    db.insert_detection(2, 2, "dog", 0.5, 0.5, 0.7, 0.9, 0.85)

    scorer = FrameScorer(db)
    # Force the scorer to pick top-2 (excluding mandatory keyframes for a clean test)
    top = scorer.score_and_select(
        candidate_frame_indices=[0, 1, 2],
        keyframe_indices=set(),
        k=2,
    )
    assert 1 in top  # frame 1 introduces the dog → must be picked
    db.close()


def test_scorer_always_includes_mandatory_keyframes(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    for idx in [0, 1, 2, 3, 4]:
        db.insert_frame(idx, idx * 1000, str(tmp_path / f"{idx}.jpg"), False, 0.0)
    # No detections at all -> scores will all be 0 except mandatory
    scorer = FrameScorer(db)
    top = scorer.score_and_select(
        candidate_frame_indices=[0, 1, 2, 3, 4],
        keyframe_indices={2, 4},
        k=3,
    )
    # Mandatory keyframes must appear regardless of K
    assert 2 in top
    assert 4 in top
    db.close()


def test_scorer_returns_sorted_indices(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    for idx in [10, 20, 30]:
        db.insert_frame(idx, idx, str(tmp_path / f"{idx}.jpg"), False, 0.0)
    scorer = FrameScorer(db)
    top = scorer.score_and_select(
        candidate_frame_indices=[10, 20, 30],
        keyframe_indices=set(),
        k=2,
    )
    assert top == sorted(top)
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_frame_scorer.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `src/modules/frame_scorer.py`**

```python
"""Score YOLO frames and pick top-K for VLM captioning."""

import logging
from typing import List, Set


class FrameScorer:
    def __init__(
        self,
        db,
        alpha: float = 1.0,   # delta weight
        beta: float = 2.0,    # detection-class set change
        gamma: float = 2.0,   # track_id churn
        delta_w: float = 0.5, # mean IoU drop
    ):
        self.db = db
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_w = delta_w
        self.logger = logging.getLogger(__name__)

    def score_and_select(
        self,
        candidate_frame_indices: List[int],
        keyframe_indices: Set[int],
        k: int,
    ) -> List[int]:
        if not candidate_frame_indices:
            return []
        sorted_candidates = sorted(candidate_frame_indices)

        # Compute score per candidate
        scored: List[tuple] = []  # (score, frame_idx)
        prev_dets = None
        prev_delta = 0.0
        for idx in sorted_candidates:
            row = self.db.get_frame(idx)
            cur_delta = row["delta_score"] if row else 0.0
            cur_dets = self.db.get_detections_for_frame(idx)

            if prev_dets is None:
                score = self.alpha * cur_delta
            else:
                set_change = self._class_set_change(prev_dets, cur_dets)
                track_churn = self._track_churn(prev_dets, cur_dets)
                iou_drop = self._mean_iou_drop(prev_dets, cur_dets)
                score = (
                    self.alpha * cur_delta
                    + self.beta * set_change
                    + self.gamma * track_churn
                    + self.delta_w * iou_drop
                )
            scored.append((score, idx))
            prev_dets = cur_dets
            prev_delta = cur_delta

        # Mandatory keyframes always selected
        forced = [idx for idx in sorted_candidates if idx in keyframe_indices]
        remaining_budget = max(0, k - len(forced))

        # Pick top remaining by score from non-forced candidates
        non_forced = [(s, i) for s, i in scored if i not in keyframe_indices]
        non_forced.sort(key=lambda x: x[0], reverse=True)
        top_non_forced = [i for _, i in non_forced[:remaining_budget]]

        result = sorted(set(forced) | set(top_non_forced))
        self.logger.info(f"Frame scorer: {len(result)}/{len(sorted_candidates)} selected for VLM")
        return result

    def _class_set_change(self, prev_dets, cur_dets) -> float:
        prev_classes = {d["class_name"] for d in prev_dets}
        cur_classes = {d["class_name"] for d in cur_dets}
        return float(len(prev_classes ^ cur_classes))

    def _track_churn(self, prev_dets, cur_dets) -> float:
        prev_ids = {d["track_id"] for d in prev_dets if d["track_id"] is not None}
        cur_ids = {d["track_id"] for d in cur_dets if d["track_id"] is not None}
        return float(len(prev_ids ^ cur_ids))

    def _mean_iou_drop(self, prev_dets, cur_dets) -> float:
        prev_by_track = {d["track_id"]: d for d in prev_dets if d["track_id"] is not None}
        cur_by_track = {d["track_id"]: d for d in cur_dets if d["track_id"] is not None}
        common = set(prev_by_track) & set(cur_by_track)
        if not common:
            return 0.0
        ious = [self._iou(prev_by_track[t], cur_by_track[t]) for t in common]
        return float(1.0 - (sum(ious) / len(ious)))

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - inter
        return inter / union if union > 0 else 0.0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_frame_scorer.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/modules/frame_scorer.py tests/test_frame_scorer.py
git commit -m "v2: frame scorer with detection-set + track-churn + IoU signals"
```

---

## Task 8: Chunked VLM captioning

**Files:**
- Modify: `src/backends/llama_server_backend.py`
- Test: `tests/test_chunked_vlm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunked_vlm.py`:

```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from src.storage import TempoGraphDB
from src.backends.llama_server_backend import LlamaServerBackend


def _jpg(p: Path):
    cv2.imwrite(str(p), np.full((100, 100, 3), 200, dtype=np.uint8))


def test_caption_chunks_propagates_seed(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    paths = []
    for i in range(20):
        p = tmp_path / f"f{i}.jpg"
        _jpg(p)
        db.insert_frame(i, i * 500, str(p), False, 0.0)
        paths.append(p)

    fake_responses = [
        {"message": {"content": "FRAME_0: a starts\nFRAME_1: a continues\nSUMMARY: agent A walking right"}},
        {"message": {"content": "FRAME_10: b appears\nSUMMARY: agent B has joined"}},
    ]
    with patch("src.backends.llama_server_backend.requests.post") as mock_post:
        mock_post.side_effect = [MagicMock(json=MagicMock(return_value=r), raise_for_status=MagicMock()) for r in fake_responses]

        backend = LlamaServerBackend()
        chunks = [
            (0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (1, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        ]
        results = backend.caption_chunks(chunks=chunks, db=db)

    assert len(results) == 2
    assert results[0].summary.startswith("agent A")
    assert results[1].summary.startswith("agent B")

    # Second call must include the first call's summary as the seed
    second_call_payload = mock_post.call_args_list[1].kwargs["json"]
    second_prompt = second_call_payload["messages"][0]["content"]
    assert "agent A walking right" in second_prompt


def test_caption_chunks_handles_failure_gracefully(tmp_path):
    db = TempoGraphDB(tmp_path / "t.db")
    p = tmp_path / "f.jpg"
    _jpg(p)
    db.insert_frame(0, 0, str(p), False, 0.0)

    with patch("src.backends.llama_server_backend.requests.post") as mock_post:
        mock_post.side_effect = Exception("boom")
        backend = LlamaServerBackend()
        results = backend.caption_chunks(chunks=[(0, [0])], db=db)

    assert len(results) == 1
    assert results[0].summary == ""  # empty seed on failure
    assert results[0].per_frame_lines == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_chunked_vlm.py -v
```
Expected: AttributeError — `caption_chunks` not defined.

- [ ] **Step 3: Add `caption_chunks` to `src/backends/llama_server_backend.py`**

Add to the imports at the top:

```python
import re
from typing import Dict, List, Tuple
from src.models import ChunkCaption
```

Add the method to the `LlamaServerBackend` class:

```python
    CHUNK_PROMPT_TEMPLATE = """You are watching a short segment of a video. Describe what is happening across these frames in chronological order.

Previous segment summary: {seed}

For each frame below, output ONE LINE describing the action. If consecutive frames show no significant change, write "(no change)". End with ONE LINE summarizing this segment in <= 20 words for use as context in the next segment.

Frame data:
{frame_block}

Output format:
FRAME_<idx>: <description>
...
SUMMARY: <one-line segment summary>
"""

    def caption_chunks(
        self,
        chunks: List[Tuple[int, List[int]]],
        db,
    ) -> List[ChunkCaption]:
        """Caption each chunk; previous chunk's summary becomes the seed for the next."""
        seed = "this is the start"
        results: List[ChunkCaption] = []

        for chunk_id, frame_indices in chunks:
            try:
                images_b64 = []
                frame_lines = []
                for fidx in frame_indices:
                    frow = db.get_frame(fidx)
                    if not frow:
                        continue
                    images_b64.append(self._encode_image(Path(frow["image_path"])))
                    dets = db.get_detections_for_frame(fidx)
                    det_text = self._format_detections(dets)
                    ts = self._format_timestamp_ms(frow["timestamp_ms"])
                    frame_lines.append(f"[frame {fidx} — t={ts}] YOLO: {det_text}")

                prompt = self.CHUNK_PROMPT_TEMPLATE.format(
                    seed=seed, frame_block="\n".join(frame_lines)
                )
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt, "images": images_b64}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                }
                response = requests.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=600
                )
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "")
                per_frame, summary = self._parse_chunk_response(content, frame_indices)

                results.append(
                    ChunkCaption(
                        chunk_id=chunk_id,
                        frame_indices=list(frame_indices),
                        per_frame_lines=per_frame,
                        summary=summary,
                        raw_response=content,
                    )
                )
                if summary:
                    seed = summary
            except Exception as e:
                self.logger.warning(f"Chunk {chunk_id} failed: {e}")
                results.append(
                    ChunkCaption(
                        chunk_id=chunk_id,
                        frame_indices=list(frame_indices),
                        per_frame_lines={},
                        summary="",
                        raw_response="",
                    )
                )

        return results

    @staticmethod
    def _format_detections(dets) -> str:
        if not dets:
            return "(none)"
        parts = []
        for d in dets:
            base = (
                f"{d['class_name']} at [{d['x1']:.2f},{d['y1']:.2f},"
                f"{d['x2']:.2f},{d['y2']:.2f}] conf={d['confidence']:.2f}"
            )
            if d.get("mean_depth") is not None:
                base += f" depth={d['mean_depth']:.2f}"
            parts.append(base)
        return "; ".join(parts)

    @staticmethod
    def _format_timestamp_ms(ms: int) -> str:
        s = ms / 1000.0
        m = int(s // 60)
        sec = s - m * 60
        return f"{m:02d}:{sec:05.2f}"

    @staticmethod
    def _parse_chunk_response(text: str, frame_indices) -> Tuple[Dict[int, str], str]:
        per_frame: Dict[int, str] = {}
        summary = ""
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"FRAME[_ ]?(\d+)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1))
                per_frame[idx] = m.group(2).strip()
                continue
            sm = re.match(r"SUMMARY\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if sm:
                summary = sm.group(1).strip()
        return per_frame, summary
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_chunked_vlm.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backends/llama_server_backend.py tests/test_chunked_vlm.py
git commit -m "v2: chunked VLM captioning with hard-wipe seed propagation"
```

---

## Task 9: Aggregator — captions to AnalysisResult

**Files:**
- Create: `src/aggregator.py`
- Test: `tests/test_aggregator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregator.py`:

```python
from unittest.mock import patch, MagicMock

from src.aggregator import CaptionAggregator
from src.models import ChunkCaption


VALID_JSON = """{
  "entities": [{"id":"E1","type":"person","description":"a man","first_seen":"00:00","last_seen":"00:10"}],
  "visual_events": [{"type":"walking","entities":["E1"],"start_time":"00:00","end_time":"00:05","description":"walks left","confidence":0.8}],
  "audio_events": [],
  "multimodal_correlations": [],
  "summary": "A man walks across the scene."
}"""


def test_aggregator_single_pass_returns_analysis_result():
    chunks = [
        ChunkCaption(0, [0, 1], {0: "man enters", 1: "man walks"}, "a man enters and walks", ""),
        ChunkCaption(1, [10], {10: "man exits"}, "the man exits", ""),
    ]
    with patch("src.aggregator.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"message": {"content": VALID_JSON}}),
            raise_for_status=MagicMock(),
        )
        agg = CaptionAggregator()
        result = agg.aggregate(chunks)

    assert len(result.entities) == 1
    assert result.entities[0].id == "E1"
    assert "man" in result.summary.lower()


def test_aggregator_falls_back_on_bad_json():
    chunks = [ChunkCaption(0, [0], {0: "blah"}, "", "")]
    with patch("src.aggregator.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"message": {"content": "not json at all"}}),
            raise_for_status=MagicMock(),
        )
        agg = CaptionAggregator()
        result = agg.aggregate(chunks)

    # Lenient parser returns empty AnalysisResult rather than crashing
    assert result.entities == []
    assert result.visual_events == []


def test_hierarchical_path_used_for_long_chunk_lists():
    long_chunks = [
        ChunkCaption(i, [i], {i: f"line {i}"}, f"sum {i}", "")
        for i in range(45)
    ]
    with patch("src.aggregator.requests.post") as mock_post:
        # First N calls compress meta-captions, last call returns final JSON
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={"message": {"content": VALID_JSON}}),
            raise_for_status=MagicMock(),
        )
        agg = CaptionAggregator(single_pass_max_chunks=30, group_size=10)
        result = agg.aggregate(long_chunks)

    assert mock_post.call_count >= 2  # at least one meta + one final
    assert result.summary != ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_aggregator.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `src/aggregator.py`**

```python
"""Aggregates per-chunk captions into an AnalysisResult."""

import logging
from typing import List

import requests

from src.json_parser import JSONParser
from src.models import AnalysisResult, ChunkCaption


SINGLE_PASS_PROMPT = """You are given a chronological log of per-frame and per-chunk descriptions of a video. Identify entities (people, animals, vehicles, objects), their behaviors and interactions over time, and produce structured JSON.

Schema:
{{"entities":[{{"id":"E1","type":"person","description":"...","first_seen":"MM:SS","last_seen":"MM:SS"}}],
"visual_events":[{{"type":"walking","entities":["E1"],"start_time":"MM:SS","end_time":"MM:SS","description":"...","confidence":0.8}}],
"audio_events":[],"multimodal_correlations":[],"summary":"..."}}

Valid behavior types: approach, depart, interact, follow, idle, group, avoid, chase, observe, moving, walking, running, standing, sitting, playing, jumping, other.

Caption log:
{captions}

Output ONLY the JSON.
"""

META_PROMPT = """Compress the following sequence of segment summaries into ONE paragraph that preserves who/what/when. Keep timestamps if present.

{block}

Output the compressed paragraph only.
"""


class CaptionAggregator:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:4b",
        single_pass_max_chunks: int = 30,
        group_size: int = 10,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.single_pass_max_chunks = single_pass_max_chunks
        self.group_size = group_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parser = JSONParser()
        self.logger = logging.getLogger(__name__)

    def aggregate(self, chunks: List[ChunkCaption]) -> AnalysisResult:
        if not chunks:
            return AnalysisResult(summary="No captions produced.")

        if len(chunks) <= self.single_pass_max_chunks:
            return self._single_pass(chunks)

        meta = self._compress_hierarchical(chunks)
        return self._single_pass_from_text(meta)

    def _single_pass(self, chunks: List[ChunkCaption]) -> AnalysisResult:
        log_lines = []
        for c in chunks:
            for fidx in c.frame_indices:
                line = c.per_frame_lines.get(fidx, "")
                if line:
                    log_lines.append(f"[frame {fidx}] {line}")
            if c.summary:
                log_lines.append(f"[chunk {c.chunk_id} summary] {c.summary}")
        return self._single_pass_from_text("\n".join(log_lines))

    def _single_pass_from_text(self, captions_text: str) -> AnalysisResult:
        prompt = SINGLE_PASS_PROMPT.format(captions=captions_text)
        response_text = self._call_ollama_text(prompt)
        return self.parser.parse(response_text)

    def _compress_hierarchical(self, chunks: List[ChunkCaption]) -> str:
        groups = [
            chunks[i : i + self.group_size]
            for i in range(0, len(chunks), self.group_size)
        ]
        meta_pieces: List[str] = []
        for grp in groups:
            block_lines = []
            for c in grp:
                if c.summary:
                    block_lines.append(f"[chunk {c.chunk_id}] {c.summary}")
            block = "\n".join(block_lines)
            meta_pieces.append(self._call_ollama_text(META_PROMPT.format(block=block)))
        return "\n".join(meta_pieces)

    def _call_ollama_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=600
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            self.logger.warning(f"Ollama text call failed: {e}")
            return ""
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_aggregator.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/aggregator.py tests/test_aggregator.py
git commit -m "v2: caption aggregator with single-pass + hierarchical paths"
```

---

## Task 10: Pipeline v2 orchestrator

**Files:**
- Create: `src/pipeline_v2.py`
- Test: `tests/test_pipeline_v2.py`

- [ ] **Step 1: Write the failing test (smoke + wiring)**

Create `tests/test_pipeline_v2.py`:

```python
from unittest.mock import patch, MagicMock
from pathlib import Path

import cv2
import numpy as np

from src.models import PipelineConfig, AnalysisResult, CameraMode
from src.pipeline_v2 import PipelineV2


def _make_video(path: Path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), 50 if i < 30 else 200, dtype=np.uint8)
        w.write(frame)
    w.release()


def test_pipeline_v2_runs_end_to_end_with_mocked_models(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video)
    out = tmp_path / "out"

    config = PipelineConfig(
        backend="llama-server",
        modules={"behavior": True, "detection": True, "depth": False, "audio": False},
        fps=1.0,
        max_frames=20,
        confidence=0.5,
        video_path=str(video),
        output_dir=str(out),
    )

    with patch("src.pipeline_v2.ObjectDetector") as MockDet, \
         patch("src.pipeline_v2.LlamaServerBackend") as MockLLM, \
         patch("src.pipeline_v2.CaptionAggregator") as MockAgg:
        MockDet.return_value.detect_to_db = MagicMock()
        MockDet.return_value.cleanup = MagicMock()
        MockLLM.return_value.caption_chunks = MagicMock(return_value=[])
        MockAgg.return_value.aggregate = MagicMock(
            return_value=AnalysisResult(summary="ok")
        )

        pipeline = PipelineV2(
            config,
            camera_mode=CameraMode.STATIC,
            yolo_fps=1.0,
            vlm_fps=0.5,
            chunk_size=10,
            depth_enabled=False,
            use_segmentation=False,
        )
        result = pipeline.run()

    assert result.analysis.summary == "ok"
    assert (out / "tempograph.db").exists()
    MockDet.return_value.detect_to_db.assert_called_once()
    MockLLM.return_value.caption_chunks.assert_called_once()
    MockAgg.return_value.aggregate.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline_v2.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `src/pipeline_v2.py`**

```python
"""TempoGraph v2 orchestrator: chunked VLM pipeline backed by SQLite."""

import gc
import logging
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import torch

from src.aggregator import CaptionAggregator
from src.backends.llama_server_backend import LlamaServerBackend
from src.graph_builder import GraphBuilder
from src.models import (
    AnalysisResult,
    CameraMode,
    PipelineConfig,
    PipelineResult,
)
from src.modules.depth import DepthEstimator
from src.modules.detector import ObjectDetector
from src.modules.frame_scorer import FrameScorer
from src.modules.frame_selector import FrameSelector
from src.storage import TempoGraphDB


class PipelineV2:
    def __init__(
        self,
        config: PipelineConfig,
        camera_mode: CameraMode = CameraMode.STATIC,
        yolo_fps: float = 1.0,
        vlm_fps: float = 0.5,
        chunk_size: int = 10,
        depth_enabled: bool = False,
        use_segmentation: bool = False,
        threshold_mult: float = 1.0,
        jpeg_quality: int = 80,
        frame_max_width: int = 640,
    ):
        self.config = config
        self.camera_mode = camera_mode
        self.yolo_fps = yolo_fps
        self.vlm_fps = vlm_fps
        self.chunk_size = chunk_size
        self.depth_enabled = depth_enabled
        self.use_segmentation = use_segmentation
        self.threshold_mult = threshold_mult
        self.jpeg_quality = jpeg_quality
        self.frame_max_width = frame_max_width
        self.logger = logging.getLogger(__name__)

    def run(self) -> PipelineResult:
        start = time.time()
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        db = TempoGraphDB(out_dir / "tempograph.db")
        try:
            # Stage 1: frame selection + JPEG export
            selection = FrameSelector().select(
                video_path=self.config.video_path,
                camera_mode=self.camera_mode,
                sample_fps=self.yolo_fps,
                threshold_mult=self.threshold_mult,
            )
            frame_paths = self._extract_and_save_frames(
                selection.frame_indices, out_dir / "frames"
            )
            video_fps, w, h = self._video_meta(self.config.video_path)
            for idx, path in zip(selection.frame_indices, frame_paths):
                ts_ms = int(idx * 1000.0 / max(video_fps, 1.0))
                db.insert_frame(
                    frame_idx=idx,
                    timestamp_ms=ts_ms,
                    image_path=str(path),
                    is_keyframe=(idx in set(selection.keyframe_indices)),
                    delta_score=self._delta_for_index(selection, idx),
                )

            # Stage 2: YOLO sweep
            model_path = "yolo11n-seg.pt" if self.use_segmentation else "yolo11n.pt"
            detector = ObjectDetector(
                model_path=model_path,
                confidence=self.config.confidence,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            detector.detect_to_db(
                frame_indices=selection.frame_indices,
                frame_paths=frame_paths,
                db=db,
                frame_width=w,
                frame_height=h,
            )
            detector.cleanup()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: depth (optional)
            if self.depth_enabled:
                depth = DepthEstimator(
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                depth.estimate_to_db(
                    frame_indices=selection.frame_indices,
                    frame_paths=frame_paths,
                    db=db,
                    output_dir=str(out_dir / "depth"),
                )
                depth.cleanup()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Stage 4: VLM frame subset
            video_duration = max(1.0, self._video_duration(self.config.video_path))
            k = max(1, int(round(video_duration * self.vlm_fps)))
            scorer = FrameScorer(db)
            vlm_frames = scorer.score_and_select(
                candidate_frame_indices=selection.frame_indices,
                keyframe_indices=set(selection.keyframe_indices),
                k=k,
            )

            # Stage 5: chunked VLM
            chunks: List[Tuple[int, List[int]]] = []
            for i in range(0, len(vlm_frames), self.chunk_size):
                chunks.append((len(chunks), vlm_frames[i : i + self.chunk_size]))
            backend = LlamaServerBackend()
            chunk_caps = backend.caption_chunks(chunks=chunks, db=db)

            # Stage 6: aggregate
            analysis = CaptionAggregator().aggregate(chunk_caps)

            # Build graph + persist
            graph = GraphBuilder()
            graph.build(analysis)
            try:
                graph.to_pyvis_html(str(out_dir / "graph.html"))
            except Exception as e:
                self.logger.warning(f"graph html failed: {e}")
            with open(out_dir / "analysis.json", "w") as f:
                f.write(analysis.model_dump_json(indent=2))

            elapsed = time.time() - start
            return PipelineResult(
                analysis=analysis,
                detection=None,
                depth=None,
                config=self.config,
                annotated_video_path=None,
                processing_time=elapsed,
            )
        finally:
            db.close()

    def _video_meta(self, path: str) -> Tuple[float, int, int]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return fps, w, h

    def _video_duration(self, path: str) -> float:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n / fps if fps > 0 else 0.0

    def _delta_for_index(self, selection, idx) -> float:
        try:
            i = selection.scan_indices.index(idx)
            return float(selection.deltas[i])
        except ValueError:
            return 0.0

    def _extract_and_save_frames(self, indices, out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(self.config.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        scale = self.frame_max_width / width if width > self.frame_max_width else 1.0
        paths: List[Path] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if scale < 1.0:
                new_w = int(width * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            p = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            paths.append(p)
        cap.release()
        return paths
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline_v2.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_v2.py tests/test_pipeline_v2.py
git commit -m "v2: PipelineV2 orchestrator wiring all six stages"
```

---

## Task 11: UI — frame selection preview (delta plot)

**Files:**
- Modify: `ui/app.py`

- [ ] **Step 1: Sketch the new control panel & preview button**

Replace the contents of `ui/app.py` with the following (preserving the imports/structure already present and only changing controls + run path; old code blocks for tabs are simplified to keep the patch reviewable):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile

import plotly.graph_objects as go
import streamlit as st

from src.models import CameraMode, PipelineConfig
from src.modules.frame_selector import FrameSelector
from src.pipeline_v2 import PipelineV2


def main():
    st.set_page_config(page_title="TempoGraph v2", layout="wide")
    st.title("TempoGraph v2 - Chunked VLM Pipeline")

    st.sidebar.header("Pipeline Configuration")

    camera_mode_label = st.sidebar.radio(
        "Camera type",
        options=["Static / fixed (CCTV)", "Moving / handheld", "Auto-detect"],
        index=0,
    )
    camera_mode = {
        "Static / fixed (CCTV)": CameraMode.STATIC,
        "Moving / handheld": CameraMode.MOVING,
        "Auto-detect": CameraMode.AUTO,
    }[camera_mode_label]

    st.sidebar.subheader("Object Detection (YOLO)")
    yolo_enabled = st.sidebar.checkbox("Enable", value=True, key="yolo_en")
    yolo_fps = st.sidebar.slider("Sweep FPS", 0.25, 4.0, 1.0, 0.25)
    use_seg = st.sidebar.checkbox("Use segmentation variant (yolo11n-seg)", value=False)
    confidence = st.sidebar.slider("Confidence", 0.1, 0.9, 0.5, 0.05)

    st.sidebar.subheader("Depth Estimation")
    depth_enabled = st.sidebar.checkbox(
        "Enable (spatial awareness — slower)", value=False
    )

    st.sidebar.subheader("VLM Captioning (llama-server)")
    vlm_fps = st.sidebar.slider("Caption FPS", 0.1, 2.0, 0.5, 0.1)
    chunk_size = st.sidebar.slider("Frames per chunk", 4, 16, 10, 1)

    st.sidebar.subheader("Frame Selection")
    threshold_mult = st.sidebar.slider("Keyframe threshold (× σ)", 0.5, 3.0, 1.0, 0.1)

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(uploaded.read())
        video_path = f.name

    col1, col2 = st.columns(2)
    if col1.button("Preview frame selection"):
        _render_selection_preview(video_path, camera_mode, yolo_fps, threshold_mult)

    if col2.button("Run full pipeline", type="primary"):
        _run_pipeline(
            video_path=video_path,
            video_name=uploaded.name,
            camera_mode=camera_mode,
            yolo_fps=yolo_fps,
            vlm_fps=vlm_fps,
            chunk_size=chunk_size,
            confidence=confidence,
            depth_enabled=depth_enabled,
            use_seg=use_seg,
            threshold_mult=threshold_mult,
        )


def _render_selection_preview(video_path, camera_mode, sample_fps, threshold_mult):
    selector = FrameSelector()
    result = selector.select(
        video_path=video_path,
        camera_mode=camera_mode,
        sample_fps=sample_fps,
        threshold_mult=threshold_mult,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.scan_indices,
            y=result.deltas,
            mode="lines",
            name="delta",
            line=dict(color="lightgray"),
        )
    )
    fig.add_hline(
        y=result.threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"threshold={result.threshold:.2f}",
    )

    kf_set = set(result.keyframe_indices)
    smp_set = set(result.sampled_indices) - kf_set

    kf_x = [i for i in result.scan_indices if i in kf_set]
    kf_y = [
        result.deltas[result.scan_indices.index(i)] for i in kf_x
    ]
    fig.add_trace(
        go.Scatter(x=kf_x, y=kf_y, mode="markers", name="keyframes (mandatory)",
                   marker=dict(color="green", size=9))
    )

    smp_x = [i for i in result.scan_indices if i in smp_set]
    smp_y = [result.deltas[result.scan_indices.index(i)] for i in smp_x]
    fig.add_trace(
        go.Scatter(x=smp_x, y=smp_y, mode="markers", name=f"sampled @ {sample_fps} Hz",
                   marker=dict(color="orange", size=8))
    )
    fig.update_layout(
        title=f"Frame selection preview ({result.camera_mode.value})",
        xaxis_title="frame index",
        yaxis_title="delta",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Total frames to process: {len(result.frame_indices)} "
        f"({len(result.keyframe_indices)} green + {len(result.sampled_indices)} orange, "
        f"after dedup)"
    )


def _run_pipeline(
    video_path,
    video_name,
    camera_mode,
    yolo_fps,
    vlm_fps,
    chunk_size,
    confidence,
    depth_enabled,
    use_seg,
    threshold_mult,
):
    config = PipelineConfig(
        backend="llama-server",
        modules={"behavior": True, "detection": True, "depth": depth_enabled, "audio": False},
        fps=yolo_fps,
        max_frames=999,
        confidence=confidence,
        video_path=video_path,
        output_dir=f"results/{video_name}",
    )
    with st.status("Running v2 pipeline...", expanded=True) as status:
        try:
            pipe = PipelineV2(
                config,
                camera_mode=camera_mode,
                yolo_fps=yolo_fps,
                vlm_fps=vlm_fps,
                chunk_size=chunk_size,
                depth_enabled=depth_enabled,
                use_segmentation=use_seg,
                threshold_mult=threshold_mult,
            )
            result = pipe.run()
            status.update(label="Done", state="complete")
            st.success(f"Done in {result.processing_time:.1f}s")
            if result.analysis:
                st.subheader("Summary")
                st.write(result.analysis.summary or "(empty)")
                st.subheader("Entities")
                st.dataframe([
                    {"id": e.id, "type": e.type, "description": e.description}
                    for e in result.analysis.entities
                ])
                st.subheader("Visual events")
                st.dataframe([
                    {"type": e.type.value if hasattr(e.type, "value") else str(e.type),
                     "entities": ", ".join(e.entities), "start": e.start_time,
                     "end": e.end_time, "description": e.description}
                    for e in result.analysis.visual_events
                ])
        except Exception as e:
            status.update(label=f"Failed: {e}", state="error")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke check (no automated test for Streamlit)**

Run:
```bash
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```
Upload any short mp4. Click "Preview frame selection" — confirm a delta plot renders with green/orange dots and a red dashed threshold line. Do NOT click "Run full pipeline" yet (Ollama may not be up).

- [ ] **Step 3: Commit**

```bash
git add ui/app.py
git commit -m "v2: Streamlit UI with camera-mode selector, separate FPS, delta preview"
```

---

## Task 12: Wire CLI entrypoint to v2 pipeline

**Files:**
- Modify: `src/pipeline_v2.py` (add CLI block at bottom)

- [ ] **Step 1: Add a CLI entry point**

Append to the end of `src/pipeline_v2.py`:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TempoGraph v2 pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/v2_run")
    parser.add_argument("--camera", default="static",
                        choices=["static", "moving", "auto"])
    parser.add_argument("--yolo-fps", type=float, default=1.0)
    parser.add_argument("--vlm-fps", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--seg", action="store_true",
                        help="Use yolo11n-seg.pt instead of yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--threshold-mult", type=float, default=1.0)
    args = parser.parse_args()

    config = PipelineConfig(
        backend="llama-server",
        modules={"behavior": True, "detection": True, "depth": args.depth, "audio": False},
        fps=args.yolo_fps,
        max_frames=999,
        confidence=args.confidence,
        video_path=args.video,
        output_dir=args.output,
    )
    pipe = PipelineV2(
        config,
        camera_mode=CameraMode(args.camera),
        yolo_fps=args.yolo_fps,
        vlm_fps=args.vlm_fps,
        chunk_size=args.chunk_size,
        depth_enabled=args.depth,
        use_segmentation=args.seg,
        threshold_mult=args.threshold_mult,
    )
    result = pipe.run()
    print(f"Done in {result.processing_time:.1f}s -> {args.output}")
```

- [ ] **Step 2: Run a syntax check**

```bash
python -c "import src.pipeline_v2"
```
Expected: no output, no error.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_v2.py
git commit -m "v2: CLI entrypoint for pipeline_v2"
```

---

## Task 13: Full test sweep + cleanup

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all pass. If pre-existing tests fail due to model imports, mark with `@pytest.mark.skip(reason="requires GPU/Ollama")` rather than disabling.

- [ ] **Step 2: Verify no orphaned imports**

```bash
python -c "from src.pipeline_v2 import PipelineV2; from src.aggregator import CaptionAggregator; from src.modules.frame_selector import FrameSelector; from src.modules.frame_scorer import FrameScorer; from src.storage import TempoGraphDB; print('imports ok')"
```
Expected: `imports ok`

- [ ] **Step 3: Commit any cleanup**

```bash
git add -A
git diff --cached --stat
git commit -m "v2: test cleanup" || echo "nothing to commit"
```

---

## Spec Coverage Self-Review

| Spec section | Implemented in |
|---|---|
| §3.1 static delta | Task 3 (`_scan_pixel_deltas`) |
| §3.2 moving / homography | Task 3 (`_scan_motion_compensated_deltas`) + Task 4 test |
| §3.3 auto-detect | Task 3 (`_auto_detect_mode`) |
| §3.4 delta plot UI | Task 11 (`_render_selection_preview`) |
| §4 YOLO sweep + DB | Task 5 (`detect_to_db`) |
| §4.3 SQLite schema | Task 2 |
| §5 depth optional + per-bbox | Task 6 (`estimate_to_db`) + Task 11 toggle |
| §6 frame scorer top-K | Task 7 |
| §7 chunked VLM + hard wipe | Task 8 |
| §8 aggregation single + hierarchical | Task 9 |
| §9 UI controls | Task 11 |
| §10 component boundaries | Tasks 2, 3, 5, 6, 7, 8, 9, 10 (file map matches) |
| §11 error handling (per-chunk failure tolerance) | Task 8 (try/except per chunk) |
| §12 testing | All TDD tests above |

---

## Notes for the Implementer

- The motion-compensated test (Task 4) can be flaky on synthetic noise — if it fails, raise `orb_features` to 1000 and ensure a textured frame (random noise) is used, not solid colors.
- The frame_extractor.py file is left untouched intentionally; v2 uses the new `frame_selector.py`. Removing the old file is out of scope.
- Existing `pipeline.py` is untouched. Switching the API/UI fully to v2 is gated on the new tests passing end-to-end.
- All llama-server calls have a 600-second timeout — if your local Ollama is slow on a long chunk, adjust `timeout=` in both `llama_server_backend.py` and `aggregator.py`.
