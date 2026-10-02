"""
Pose estimation (dead reckoning) and waypoint following for the Tello.

The Tello has no absolute positioning, but its 'go x y z speed' command uses
the downward vision/optical-flow sensor and is fairly accurate. We keep track
of the position by adding up every commanded move. While a move is running
the position is interpolated over time, so puddles seen mid-flight still get
a reasonable coordinate.
"""

import math
import threading
import time

from . import config as cfg


class MissionAborted(Exception):
    pass


class Pose:
    """Thread-safe position estimate in the mission frame (cm, deg)."""

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self._lock = threading.Lock()
        self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), float(yaw)
        self._seg = None  # (start(x,y,z), end(x,y,z), t0, duration)

    def set(self, x=None, y=None, z=None, yaw=None):
        with self._lock:
            self._seg = None
            if x is not None:
                self.x = float(x)
            if y is not None:
                self.y = float(y)
            if z is not None:
                self.z = float(z)
            if yaw is not None:
                self.yaw = float(yaw)

    def begin_move(self, target, speed):
        with self._lock:
            start = (self.x, self.y, self.z)
            dist = math.dist(start, target)
            self._seg = (start, tuple(target), time.time(), max(dist / speed, 0.1))

    def end_move(self):
        with self._lock:
            if self._seg:
                self.x, self.y, self.z = self._seg[1]
                self._seg = None

    def cancel_move(self):
        """Move failed: keep the interpolated position as best guess."""
        x, y, z, _ = self.get()
        with self._lock:
            self.x, self.y, self.z = x, y, z
            self._seg = None

    def get(self, t=None):
        """Return (x, y, z, yaw), interpolated at time t (default: now)."""
        with self._lock:
            if self._seg is None:
                return self.x, self.y, self.z, self.yaw
            start, end, t0, duration = self._seg
            t = time.time() if t is None else t
            f = min(max((t - t0) / duration, 0.0), 1.0)
            p = [s + (e - s) * f for s, e in zip(start, end)]
            return p[0], p[1], p[2], self.yaw

    def world_to_body(self, dx, dy):
        """Rotate a mission-frame delta into the drone body frame (fwd, left)."""
        yaw = math.radians(self.yaw)
        bx = dx * math.cos(yaw) + dy * math.sin(yaw)
        by = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return bx, by


def validate_waypoints(raw):
    """Convert the Jetson's waypoint list into [(x, y, z), ...] and check the geofence."""
    wps = []
    for i, wp in enumerate(raw):
        if isinstance(wp, dict):
            x, y, z = wp["x"], wp["y"], wp["z"]
        else:
            x, y, z = wp
        x, y, z = float(x), float(y), float(z)
        for name, val in (("x", x), ("y", y), ("z", z)):
            lo, hi = cfg.GEOFENCE[name]
            if not lo <= val <= hi:
                raise ValueError(f"waypoint {i}: {name}={val} outside geofence [{lo}, {hi}]")
        wps.append((x, y, z))
    if not wps:
        raise ValueError("mission has no waypoints")
    return wps


class PathFollower:
    def __init__(self, tello, pose, abort_event):
        self.tello = tello
        self.pose = pose
        self.abort = abort_event
        if cfg.MAX_STEP_CM < 40 or cfg.MAX_STEP_CM > 500:
            raise ValueError("MAX_STEP_CM must be between 40 and 500")

    def fly(self, waypoints, speed, on_waypoint=None):
        """Fly through all waypoints. Raises MissionAborted when abort is set."""
        speed = int(min(max(speed, 10), 100))
        for i, wp in enumerate(waypoints):
            self.move_to(wp, speed)
            if on_waypoint:
                on_waypoint(i, wp)

    def move_to(self, target, speed):
        while True:
            if self.abort.is_set():
                raise MissionAborted()
            x, y, z, _ = self.pose.get()
            dx, dy, dz = target[0] - x, target[1] - y, target[2] - z
            # The Tello can't do moves where every axis is < 20 cm. The remaining
            # error is not lost: the pose keeps the real position, so the next
            # waypoint corrects for it.
            bx, by = self.pose.world_to_body(dx, dy)
            if max(abs(bx), abs(by), abs(dz)) < cfg.MIN_MOVE_CM:
                return
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            scale = min(1.0, cfg.MAX_STEP_CM / dist)
            sdx, sdy, sdz = dx * scale, dy * scale, dz * scale
            bx, by = self.pose.world_to_body(sdx, sdy)
            bx, by, bz = int(round(bx)), int(round(by)), int(round(sdz))
            # Store the rounded move so the pose matches what the drone really does
            yaw = math.radians(self.pose.yaw)
            wdx = bx * math.cos(yaw) - by * math.sin(yaw)
            wdy = bx * math.sin(yaw) + by * math.cos(yaw)
            self.pose.begin_move((x + wdx, y + wdy, z + bz), speed)
            try:
                self.tello.go_xyz_speed(bx, by, bz, speed)
            except Exception:
                self.pose.cancel_move()
                raise
            self.pose.end_move()
