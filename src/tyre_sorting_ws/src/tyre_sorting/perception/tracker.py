"""Lightweight frame-to-frame centroid tracker for conveyor fragments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .segmentation_node import FragmentDetection

@dataclass
class Track:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    last_time: float
    missed: int = 0

class CentroidTracker:
    def __init__(self, max_distance: float = 0.12, max_missed: int = 5):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[FragmentDetection], dt: float, points_3d: list[np.ndarray]):
        if not points_3d:
            for t in self.tracks.values(): t.missed += 1
            self._prune(); return []
        unmatched = set(range(len(points_3d)))
        assignments = []
        for tid, tr in list(self.tracks.items()):
            pred = tr.position + tr.velocity * dt
            if unmatched:
                j = min(unmatched, key=lambda k: np.linalg.norm(points_3d[k] - pred))
                dist = np.linalg.norm(points_3d[j] - pred)
                if dist <= self.max_distance:
                    assignments.append((tid, j)); unmatched.remove(j)
                    tr.velocity = 0.7 * tr.velocity + 0.3 * ((points_3d[j] - tr.position) / max(dt, 1e-6))
                    tr.position = points_3d[j]; tr.last_time += dt; tr.missed = 0
                else:
                    tr.missed += 1
            else:
                tr.missed += 1
        self._prune()
        for j in unmatched:
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = Track(tid, points_3d[j].copy(), np.zeros(3), 0.0)
            assignments.append((tid, j))
        assignments.sort(key=lambda x: x[1])
        return assignments

    def _prune(self):
        for tid in [k for k,v in self.tracks.items() if v.missed > self.max_missed]:
            del self.tracks[tid]
