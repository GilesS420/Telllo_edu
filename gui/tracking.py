"""Describe flight readiness and what position samples establish for the display."""

import math

from drone import config as cfg


def flip_readiness(state, airborne, motion_phase=None, battery=None, flip_busy=False):
    """Return button availability, its caption, and a rejection or warning message."""
    if state == "taking_off" or motion_phase == "takeoff_sdk":
        return False, "🔁  Salto · wacht op opstijgen", "Wacht tot de drone klaar is met opstijgen."
    if state == "landing":
        return False, "🔁  Salto · landen", "Geen salto tijdens het landen."
    if not airborne:
        return False, "🔁  Salto · eerst opstijgen", "Drone moet in de lucht zijn om een salto te doen."
    if flip_busy:
        return False, "🔁  Salto · bezig", "Er is al een salto bezig of ingepland."
    if battery is not None and battery < cfg.FLIP_MIN_BATTERY:
        return (False, f"🔁  Salto · batterij <{cfg.FLIP_MIN_BATTERY}%",
                f"Batterij te laag voor salto ({battery:.0f}% < {cfg.FLIP_MIN_BATTERY}%).")
    if battery is not None and battery < 50:
        return (True, f"🔁  Salto · {battery:.0f}% (<50%)",
                "Batterij <50%: de drone kan de salto weigeren.")
    return True, "🔁  Salto", ""


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
