"""Bounded horizontal recovery of a recently observed landing marker."""

import math


class PadRecovery:
    def __init__(self, max_speed=6.0, max_distance=40.0, timeout=3.0):
        self.max_speed, self.max_distance, self.timeout = max_speed, max_distance, timeout
        self.target = None
        self.last_seen = None
        self.scope = None
        self.initial_error = None

    @staticmethod
    def _scope(snapshot, now):
        stamp = snapshot.get("measurement_time")
        if not snapshot.get("measured") or stamp is None or not 0 <= now - stamp <= 0.35:
            return None
        if snapshot.get("position_valid"):
            return "global"
        if snapshot.get("local_reference_valid"):
            return "local"
        return None

    def observe(self, snapshot, offset, timestamp, now):
        scope = self._scope(snapshot, now)
        if scope is None or not 0 <= now - timestamp <= 0.25:
            self.target = None
            return
        if self.last_seen is not None and timestamp <= self.last_seen:
            return
        target = tuple(snapshot["pose"][i] + offset[i] for i in (0, 1))
        if not all(math.isfinite(v) for v in target):
            self.target = None
            return
        self.target, self.last_seen, self.scope = target, timestamp, scope
        self.initial_error = None

    def velocity(self, snapshot, now):
        scope = self._scope(snapshot, now)
        if (self.target is None or scope is None or scope != self.scope
                or not 0 <= now - self.last_seen <= self.timeout):
            self.target = None
            return (0.0, 0.0, 0.0)
        error = tuple(self.target[i] - snapshot["pose"][i] for i in (0, 1))
        if not all(math.isfinite(v) for v in error):
            self.target = None
            return (0.0, 0.0, 0.0)
        distance = math.hypot(*error)
        self.initial_error = distance if self.initial_error is None else self.initial_error
        if distance > self.max_distance or distance > self.initial_error + 3:
            self.target = None
            return (0.0, 0.0, 0.0)
        scale = min(0.8, self.max_speed / max(distance, 1e-6))
        return (error[0] * scale, error[1] * scale, 0.0)
