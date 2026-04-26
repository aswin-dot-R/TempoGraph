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
