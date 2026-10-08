"""
Telemetry: every sensor value of the Tello over time, plus a terrain map.

The Tello sends a state packet ~10x per second (djitellopy keeps the latest one,
reading it sends no command). We sample it, keep a few minutes of history for
the graphs in the GUI and build a terrain map from two sensors:

* barometer:  altitude of the drone (air pressure), relative to the start
* ToF sensor: distance from the drone down to whatever is below it

    ground elevation = barometer altitude - ToF distance

Flying over a box: the ToF distance drops while the altitude stays the same,
so the ground under the drone is higher. The barometer is noisy (about
±10-20 cm) and drifts slowly with the air pressure, so samples are averaged per
grid cell and the floor can be re-referenced ("Herijken" in the GUI). Good
enough to find boxes, steps and ramps, not for millimetres.
"""

import csv
import math
import threading
import time
from collections import deque

from . import config as cfg

# Sample fields, in CSV/column order
FIELDS = ("t", "flight_s", "battery", "temp", "tof", "h", "baro", "alt", "ground",
          "pitch", "roll", "yaw_imu", "vx", "vy", "vz", "ax", "ay", "az",
          "x", "y", "z", "yaw", "odo", "odo_q", "pad")


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class TerrainMap:
    """Ground elevation (cm, relative to the floor at the start) on a grid."""

    def __init__(self, cell_cm=None):
        self.cell = cell_cm or cfg.TERRAIN_CELL_CM
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.cells = {}          # (i, j) -> [n, mean, M2]
            self.profile = []        # (track distance cm, elevation cm, x, y)
            self.version = 0         # bumps on every change (the GUI redraws on change)
            self._ref = None         # altitude - tof over the floor at the start
            self._ref_samples = []
            self._last_xy = None
            self._dist = 0.0

    def rereference(self):
        """Take the ground under the drone as the new floor level (0 cm)."""
        with self._lock:
            self._ref = None
            self._ref_samples = []

    @property
    def calibrated(self):
        return self._ref is not None

    def add(self, x, y, alt_cm, tof_cm):
        """One measurement. Returns the ground elevation (cm) or None."""
        raw = alt_cm - tof_cm
        with self._lock:
            if self._ref is None:
                self._ref_samples.append(raw)
                if len(self._ref_samples) < cfg.TERRAIN_REF_SAMPLES:
                    return None
                s = sorted(self._ref_samples)
                self._ref = s[len(s) // 2]
            elev = raw - self._ref
            key = (math.floor(x / self.cell), math.floor(y / self.cell))
            c = self.cells.setdefault(key, [0, 0.0, 0.0])
            c[0] += 1                                     # Welford mean/variance
            d = elev - c[1]
            c[1] += d / c[0]
            c[2] += d * (elev - c[1])
            if self._last_xy is not None:
                self._dist += math.dist(self._last_xy, (x, y))
            self._last_xy = (x, y)
            self.profile.append((self._dist, elev, x, y))
            if len(self.profile) > cfg.TERRAIN_PROFILE_MAX:
                del self.profile[:len(self.profile) // 4]
            self.version += 1
            return elev

    def snapshot(self):
        """[(x_center, y_center, mean, std, n), ...] for every visited cell."""
        with self._lock:
            out = []
            for (i, j), (n, mean, m2) in self.cells.items():
                std = math.sqrt(m2 / (n - 1)) if n > 1 else 0.0
                out.append(((i + 0.5) * self.cell, (j + 0.5) * self.cell, mean, std, n))
            return out

    def elevation_at(self, x, y):
        with self._lock:
            c = self.cells.get((math.floor(x / self.cell), math.floor(y / self.cell)))
            return c[1] if c else None

    def profile_copy(self):
        with self._lock:
            return list(self.profile)

    def summary(self):
        """Numbers for the terrain panel."""
        cells = [c for c in self.snapshot() if c[4] >= cfg.TERRAIN_MIN_SAMPLES]
        if not cells:
            return None
        elev = [c[2] for c in cells]
        high = [c for c in cells if c[2] >= cfg.TERRAIN_OBSTACLE_CM]
        low = [c for c in cells if c[2] <= -cfg.TERRAIN_OBSTACLE_CM]
        mean = sum(elev) / len(elev)
        rough = math.sqrt(sum((e - mean) ** 2 for e in elev) / len(elev))
        return {
            "cells": len(cells), "area_m2": len(cells) * (self.cell / 100) ** 2,
            "min": min(elev), "max": max(elev), "mean": mean, "roughness": rough,
            "obstacles": cluster(high, self.cell), "dips": cluster(low, self.cell),
        }


def cluster(cells, cell_cm):
    """Group neighbouring cells -> [{x, y, size_cm2, height}, ...] (obstacles)."""
    todo = {(math.floor(c[0] / cell_cm), math.floor(c[1] / cell_cm)): c for c in cells}
    groups = []
    while todo:
        key, c = todo.popitem()
        stack, members = [key], [c]
        while stack:
            i, j = stack.pop()
            for n in ((i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                if n in todo:
                    members.append(todo.pop(n))
                    stack.append(n)
        peak = max(members, key=lambda m: abs(m[2]))
        groups.append({
            "x": sum(m[0] for m in members) / len(members),
            "y": sum(m[1] for m in members) / len(members),
            "size_cm2": len(members) * cell_cm * cell_cm,
            "height": peak[2],
        })
    return sorted(groups, key=lambda g: -abs(g["height"]))


class Telemetry:
    """Samples all Tello sensors in a background thread."""

    def __init__(self, tello, pose, is_airborne, hz=None, history_s=None):
        self.tello = tello
        self.pose = pose
        self.is_airborne = is_airborne
        self.hz = hz or cfg.TELEMETRY_HZ
        self.samples = deque(maxlen=int((history_s or cfg.TELEMETRY_HISTORY_S) * self.hz))
        self.terrain = TerrainMap()
        self._lock = threading.Lock()
        self._baro0 = None           # barometer on the ground at the start
        self._alt = None             # filtered altitude
        self.running = True
        self.latest = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def _state(self):
        try:
            return self.tello.get_current_state() or {}
        except Exception:
            return {}

    def _sample(self, now, dt):
        s = self._state()
        airborne = self.is_airborne()
        x, y, z, yaw = self.pose.get()

        tof = _num(s.get("tof"))
        if tof is not None and not cfg.TOF_MIN_CM < tof < cfg.TOF_MAX_CM:
            tof = None                       # 10 = too close / no reading
        baro = _num(s.get("baro"))
        baro = baro * 100 if baro is not None else None      # m -> cm
        h = _num(s.get("h"))

        # altitude relative to the start (barometer, or the Tello's own 'h')
        source = h if cfg.TERRAIN_ALT_SOURCE == "h" else baro
        alt = raw = None
        if source is not None:
            if self._baro0 is None:
                self._baro0 = source
            if not airborne:
                # on the ground: keep re-zeroing, so the takeoff starts at 0
                self._baro0 += 0.1 * (source - self._baro0)
            raw = source - self._baro0
            # smoothed for the display only: a filter would lag behind the ToF
            # and bias the terrain while climbing; the map averages the noise per cell
            a = min(1.0, dt / cfg.TERRAIN_ALT_TAU_S) if self._alt is not None else 1.0
            self._alt = raw if self._alt is None else self._alt + a * (raw - self._alt)
            alt = self._alt

        ground = None
        if airborne and raw is not None and tof is not None:
            ground = self.terrain.add(x, y, raw, tof)

        temp = None
        tl, th = _num(s.get("templ")), _num(s.get("temph"))
        if tl is not None and th is not None:
            temp = (tl + th) / 2
        dm = lambda k: None if _num(s.get(k)) is None else _num(s.get(k)) * 10   # dm/s -> cm/s
        g = lambda k: None if _num(s.get(k)) is None else _num(s.get(k)) / 1000  # mg -> g
        pad = _num(s.get("mid"))
        return {
            "t": now, "flight_s": _num(s.get("time")), "battery": _num(s.get("bat")),
            "temp": temp, "tof": tof, "h": h, "baro": baro, "alt": alt, "ground": ground,
            "pitch": _num(s.get("pitch")), "roll": _num(s.get("roll")),
            "yaw_imu": _num(s.get("yaw")),
            "vx": dm("vgx"), "vy": dm("vgy"), "vz": dm("vgz"),
            "ax": g("agx"), "ay": g("agy"), "az": g("agz"),
            "x": x, "y": y, "z": z, "yaw": yaw,
            "odo": bool(self.pose.measured) if airborne else None,
            "odo_q": self.pose.quality if airborne else None,
            "pad": int(pad) if pad is not None and pad > 0 else None,
        }

    def _loop(self):
        period = 1.0 / self.hz
        last = time.time()
        while self.running:
            t0 = time.time()
            try:
                sample = self._sample(t0, t0 - last)
            except Exception as e:  # never kill the thread over one bad packet
                print(f"⚠️  Telemetry: {e}")
                sample = None
            last = t0
            if sample:
                with self._lock:
                    self.samples.append(sample)
                    self.latest = sample
            time.sleep(max(0.0, period - (time.time() - t0)))

    def series(self, keys, window_s):
        """{'t': [...seconds before now (negative)], key: [...]} for the graphs."""
        now = time.time()
        with self._lock:
            rows = [s for s in self.samples if now - s["t"] <= window_s]
        out = {"t": [s["t"] - now for s in rows]}
        for k in keys:
            out[k] = [s.get(k) for s in rows]
        return out

    def export_csv(self, path):
        with self._lock:
            rows = list(self.samples)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else
                                round(r[k], 3) if isinstance(r[k], float) else r[k])
                            for k in FIELDS})
        return len(rows)

    def stop(self):
        self.running = False
