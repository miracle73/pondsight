from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List
import numpy as np


@dataclass
class Track:
    track_id: int
    lengths_px: List[float] = field(default_factory=list)
    boxes: List = field(default_factory=list)

    @property
    def mean_length_px(self) -> float:
        return float(np.mean(self.lengths_px)) if self.lengths_px else 0.0

    def add(self, box, length_px: float):
        self.boxes.append(box)
        self.lengths_px.append(length_px)


class Tracker:
    def __init__(self, smooth_window: int = 9):
        self.tracks: Dict[int, Track] = {}
        self.smooth_window = smooth_window
        self._raw_lengths: Dict[int, List[float]] = defaultdict(list)

    def update(self, results) -> Dict[int, Track]:
        if results is None or not hasattr(results, "boxes") or results.boxes is None:
            return self.tracks
        boxes = results.boxes
        if boxes.id is None:
            return self.tracks
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        for tid, box in zip(ids, xyxy):
            diag = float(np.sqrt((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2))
            self._raw_lengths[tid].append(diag)
            smoothed = float(np.median(self._raw_lengths[tid][-self.smooth_window:]))
            if tid not in self.tracks:
                self.tracks[tid] = Track(track_id=int(tid))
            self.tracks[tid].add(box, smoothed)
        return self.tracks

    @property
    def active_ids(self) -> list:
        return list(self.tracks.keys())

    def total_id_switches(self) -> int:
        return max(0, len(self.tracks) - 1) if self.tracks else 0

    def mean_track_length(self) -> float:
        if not self.tracks:
            return 0.0
        return float(np.mean([len(t.lengths_px) for t in self.tracks.values()]))
