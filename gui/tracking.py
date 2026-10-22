"""Describe what a position sample actually establishes for the display."""

import math

from drone import config as cfg


class TrajectoryHistory:
    """Keep measured path sections; unknown movement creates a gap, not a reset."""

    def __init__(self):
        self.frame = None
        self.mode = None
        self.global_points = []
        self.local_points = []
        self._break = True

    def clear(self):
        self.global_points.clear()
        self.local_points.clear()
        self._break = True

    def update(self, frame, mode, pose):
        if frame != self.frame:
            self.clear()
            self.frame = frame
        if mode != self.mode:
            self._break = True
            self.mode = mode
        if mode not in ("global", "local"):
            return
        point = tuple(pose[:3])
        if not all(math.isfinite(v) for v in point):
            self._break = True
            return
        points = self.global_points if mode == "global" else self.local_points
        if self._break or not points or max(abs(a - b) for a, b in zip(points[-1], point)) > 2:
            if self._break and points:
                points.append((math.nan, math.nan, math.nan))
            points.append(point)
        self._break = False


def display_tracking(snapshot, airborne, now):
    if not airborne:
        return "ground", ""
    stamp = snapshot.get("measurement_time")
    fresh = stamp is not None and 0 <= now - stamp <= cfg.VO_LOST_S
    if not snapshot["measured"] or not fresh:
        return "lost", ("Positie onbekend — model toont laatste schatting\n"
                        "Werkelijke verplaatsing wordt niet gemeten")
    if snapshot["position_valid"]:
        return "global", ""
    return "local", "Relatieve beweging gemeten — startpositie onbekend"
