"""Generate a synthetic fixture video for TempoGraph v2 smoke tests."""

from pathlib import Path
import sys

import cv2
import numpy as np


FPS = 30
WIDTH = 640
HEIGHT = 480
DURATION_SECONDS = 10
TOTAL_FRAMES = FPS * DURATION_SECONDS
OUTPUT_PATH = Path("tests/fixtures/sample.mp4")


def _draw_person(frame: np.ndarray, x: int, y: int, scale: float = 1.0) -> None:
    head_r = int(14 * scale)
    body_h = int(48 * scale)
    limb = int(24 * scale)
    color = (245, 245, 245)

    cv2.circle(frame, (x, y), head_r, color, -1)
    cv2.line(frame, (x, y + head_r), (x, y + head_r + body_h), color, 4)
    cv2.line(
        frame,
        (x - limb, y + head_r + 14),
        (x + limb, y + head_r + 14),
        color,
        4,
    )
    cv2.line(
        frame,
        (x, y + head_r + body_h),
        (x - limb, y + head_r + body_h + limb),
        color,
        4,
    )
    cv2.line(
        frame,
        (x, y + head_r + body_h),
        (x + limb, y + head_r + body_h + limb),
        color,
        4,
    )


def _draw_car(frame: np.ndarray, x: int, y: int, scale: float = 1.0) -> None:
    body_w = int(96 * scale)
    body_h = int(28 * scale)
    roof_w = int(52 * scale)
    roof_h = int(20 * scale)
    wheel_r = int(10 * scale)

    body_color = (55, 190, 255)
    wheel_color = (30, 30, 30)

    cv2.rectangle(frame, (x, y), (x + body_w, y + body_h), body_color, -1)
    roof_pts = np.array(
        [
            [x + 14, y],
            [x + 30, y - roof_h],
            [x + 30 + roof_w, y - roof_h],
            [x + 70, y],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, roof_pts, body_color)
    cv2.circle(frame, (x + 22, y + body_h), wheel_r, wheel_color, -1)
    cv2.circle(frame, (x + body_w - 22, y + body_h), wheel_r, wheel_color, -1)


def _scene_one(frame_idx: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (35, 55, 150), dtype=np.uint8)
    phase = frame_idx / float(FPS * 3)

    rect_x = int(40 + phase * 360)
    cv2.rectangle(frame, (rect_x, 90), (rect_x + 110, 250), (40, 220, 90), -1)
    _draw_person(frame, x=rect_x + 55, y=130, scale=1.0)

    cv2.putText(
        frame,
        "Scene 1: walking figure",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return frame


def _scene_two(frame_idx: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (80, 165, 70), dtype=np.uint8)
    local_idx = frame_idx - (FPS * 3)
    t = local_idx / float(FPS * 3)

    circle_x = int(80 + t * 460)
    circle_y = int(220 + np.sin(t * np.pi * 4) * 110)
    cv2.circle(frame, (circle_x, circle_y), 52, (255, 240, 70), -1)

    car_x = int(40 + t * 420)
    _draw_car(frame, x=car_x, y=390, scale=1.0)

    cv2.putText(
        frame,
        "Scene 2: bouncing circle + car",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
    )
    return frame


def _scene_three(frame_idx: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (150, 80, 55), dtype=np.uint8)
    local_idx = frame_idx - (FPS * 6)
    t = local_idx / float(FPS * 4)

    left_x = int(40 + t * 220)
    right_x = int(520 - t * 220)
    tri_y = int(140 + np.sin(t * np.pi * 6) * 50)
    poly = np.array(
        [[left_x, tri_y], [left_x - 40, tri_y + 90], [left_x + 40, tri_y + 90]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, poly, (95, 245, 255))

    cv2.rectangle(frame, (right_x, 260), (right_x + 90, 360), (255, 90, 180), -1)
    cv2.line(frame, (0, 420), (WIDTH, 420), (240, 240, 240), 3)
    _draw_person(frame, x=120 + int(t * 120), y=330, scale=0.85)

    cv2.putText(
        frame,
        "Scene 3: crossing motion",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return frame


def render_frame(frame_idx: int) -> np.ndarray:
    if frame_idx < FPS * 3:
        return _scene_one(frame_idx)
    if frame_idx < FPS * 6:
        return _scene_two(frame_idx)
    return _scene_three(frame_idx)


def main() -> int:
    if OUTPUT_PATH.exists():
        print(f"Fixture already exists: {OUTPUT_PATH}")
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {OUTPUT_PATH}")

    try:
        for frame_idx in range(TOTAL_FRAMES):
            writer.write(render_frame(frame_idx))
    finally:
        writer.release()

    print(f"Wrote synthetic fixture: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
