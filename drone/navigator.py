"""
Pose estimation and waypoint following for the Tello.

Pose: the position is updated ~20x per second by odometry.py with the real
movement measured by the downward camera (visual odometry), the IMU heading and
the height sensor. When the odometry has no good image (no floor texture) the
commanded moves are used instead, like plain dead reckoning.

PathFollower, two modes:
* "go": a 'go x y z' move per waypoint (split in steps of MAX_STEP_CM). Before
  every move the heading is corrected and the move is computed from the
  *measured* position, so drift while hovering and wind are corrected.
* "rc": smooth flight without stopping. The drone is steered 15x per second
  along the path (pure pursuit) and pushed back onto it when it drifts. Needs
  working odometry; falls back to "go" when the odometry is lost.
Optionally the drone positions itself precisely on a waypoint (FINE_POSITION).
"""

import collections
import math
import threading
import time

from . import config as cfg


class MissionAborted(Exception):
    pass


class OdometryLost(Exception):
    pass


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class Pose:
    """Thread-safe position estimate in the mission frame (cm, deg)."""

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self._lock = threading.Lock()
        self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), float(yaw)
        self.reference_yaw = float(yaw)  # mission frame heading at the start (held in flight)
        self._yaw_offset = None
        self._seg = None
        self._hist = collections.deque(maxlen=200)
        self.measured = False   # visual odometry is tracking right now
        self.quality = 0.0      # last phase correlation response (for the GUI)
        self.vo_disabled = False  # set when the odometry contradicts the commands
        self.vo_validated = False  # odometry matched at least one commanded move

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
                self.yaw = self.reference_yaw = float(yaw)
                self._yaw_offset = None  # re-anchored at the next IMU reading

    def update_imu(self, imu_yaw):
        """Tello IMU yaw (deg) -> mission yaw. The first reading anchors the frame."""
        with self._lock:
            if self._yaw_offset is None:
                self._yaw_offset = self.yaw - cfg.YAW_SIGN * imu_yaw
            self.yaw = wrap_deg(self._yaw_offset + cfg.YAW_SIGN * imu_yaw)

    # --- commanded moves ('go') ------------------------------------------------
    def begin_move(self, target, speed):
        with self._lock:
            start = (self.x, self.y, self.z)
            duration = max(math.dist(start, target) / speed, 0.1)
            vel = tuple((e - s) / duration for s, e in zip(start, target))
            self._seg = {"start": start, "end": tuple(target), "t0": time.time(),
                         "dur": duration, "vel": vel,
                         "measured": 0.0, "total": 0.0, "z_measured": False}

    def end_move(self):
        """Move done. Without (enough) odometry we trust the commanded target."""
        with self._lock:
            seg, self._seg = self._seg, None
            if seg is None:
                return
            if seg["total"] == 0 or seg["measured"] < 0.5 * seg["total"]:
                self.x, self.y = seg["end"][0], seg["end"][1]
            else:
                self._check_odometry(seg)
            if not seg["z_measured"]:
                self.z = seg["end"][2]

    def _check_odometry(self, seg):
        """Safety: the measured move must roughly match the commanded move."""
        cx, cy = seg["end"][0] - seg["start"][0], seg["end"][1] - seg["start"][1]
        mx, my = self.x - seg["start"][0], self.y - seg["start"][1]
        c = math.hypot(cx, cy)
        if c < 40:
            return
        along = (mx * cx + my * cy) / c  # measured distance in the commanded direction
        if 0.4 * c < along < 2.5 * c:
            self.vo_validated = True
            return
        # Never trust it again this flight: a wrong camera direction would make
        # every correction push the drone further away.
        self.vo_disabled = True
        self.measured = False
        self.x, self.y = seg["end"][0], seg["end"][1]
        print(f"⚠️  Odometrie mat {along:.0f} cm voor een beweging van {c:.0f} cm: odometrie "
              "UITGESCHAKELD. Controleer CAM_FORWARD_SIGN / CAM_LEFT_SIGN / CAM_HFOV_DEG.")

    def cancel_move(self):
        with self._lock:
            self._seg = None

    # --- odometry updates (called by odometry.py) -----------------------------
    def step(self, dt, now, vo=None, height=None, measured=False, quality=0.0):
        """
        vo: (dx, dy) displacement in cm (mission frame) measured since the last call
        height: measured height (cm) or None
        measured: odometry is tracking; if not, commanded moves are integrated
        """
        with self._lock:
            self.measured, self.quality = measured, quality
            seg = self._seg
            active = seg is not None and now < seg["t0"] + seg["dur"]
            if seg is not None:
                seg["total"] += dt
                if measured:
                    seg["measured"] += dt
            if vo is not None:
                self.x += vo[0]
                self.y += vo[1]
            elif not measured and active:
                self.x += seg["vel"][0] * dt
                self.y += seg["vel"][1] * dt
            if height is not None:
                self.z = 0.7 * self.z + 0.3 * height if self.z > 0 else height
                if seg is not None:
                    seg["z_measured"] = True
            elif active:
                self.z += seg["vel"][2] * dt
            self._hist.append((now, self.x, self.y, self.z, self.yaw))

    def get(self, t=None):
        """Return (x, y, z, yaw), now or (approximately) at an earlier time t."""
        with self._lock:
            if t is not None:
                for sample in reversed(self._hist):
                    if sample[0] <= t:
                        return sample[1:]
            return self.x, self.y, self.z, self.yaw

    # --- frames -----------------------------------------------------------------
    def _body_to_world(self, fwd, left):
        yaw = math.radians(self.yaw)
        return (fwd * math.cos(yaw) - left * math.sin(yaw),
                fwd * math.sin(yaw) + left * math.cos(yaw))

    def body_to_world(self, fwd, left):
        with self._lock:
            return self._body_to_world(fwd, left)

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
        self.yaw_hold = cfg.YAW_HOLD
        if cfg.MAX_STEP_CM < 40 or cfg.MAX_STEP_CM > 500:
            raise ValueError("MAX_STEP_CM must be between 40 and 500")

    def _check_abort(self):
        if self.abort.is_set():
            raise MissionAborted()

    # ------------------------------------------------------------------ mission
    def fly(self, waypoints, speed, on_waypoint=None, mode=None, fine=None):
        """Fly through all waypoints. Raises MissionAborted when abort is set."""
        speed = int(min(max(speed, 10), 100))
        mode = mode or cfg.NAV_MODE
        fine = fine or cfg.FINE_POSITION
        if fine == "auto":
            fine = "all" if mode == "go" else "last"
        x, y, z, _ = self.pose.get()
        prev = (x, y, z)
        if mode == "rc" and not self.pose.vo_validated:
            print("ℹ️  Eerste punt in stap-modus om de odometrie te controleren, "
                  "daarna vloeiend")
        try:
            for i, wp in enumerate(waypoints):
                last = i == len(waypoints) - 1
                # rc steering only once the odometry has been checked against
                # a normal 'go' move (the first leg is flown with 'go')
                if mode == "rc" and self.pose.vo_validated and not self.pose.vo_disabled:
                    try:
                        self.rc_leg(prev, wp, speed, stop_at_end=last)
                    except OdometryLost:
                        print("⚠️  Odometrie kwijt: verder in stap-modus")
                        self.tello.send_rc_control(0, 0, 0, 0)
                        mode = "go"
                        self.move_to(wp, speed)
                else:
                    self.move_to(wp, speed)
                if fine == "all" or (fine == "last" and last):
                    self.hold_position(wp)
                prev = wp
                if on_waypoint:
                    on_waypoint(i, wp)
        finally:
            if mode == "rc":
                self.tello.send_rc_control(0, 0, 0, 0)

    # --------------------------------------------------------------- heading
    def yaw_error(self):
        return wrap_deg(self.pose.reference_yaw - self.pose.yaw)

    def correct_yaw(self):
        """'go' mode: rotate back to the start heading when it is off too much."""
        if not self.yaw_hold:
            return
        err = self.yaw_error()
        if abs(err) <= cfg.YAW_TOL_DEG:
            return
        deg = int(round(abs(err)))
        print(f"🧭 Richting corrigeren: {err:+.0f}°")
        if err > 0:   # mission yaw is counter-clockwise positive
            self.tello.rotate_counter_clockwise(deg)
        else:
            self.tello.rotate_clockwise(deg)
        time.sleep(0.3)  # let the IMU reading catch up
        new_err = self.yaw_error()
        if abs(new_err) > abs(err) + 2:
            # Rotating made it worse: the yaw sign is wrong. Never keep spinning.
            self.yaw_hold = False
            print("⚠️  Richting werd slechter na correctie: richting vasthouden UIT. "
                  "Controleer YAW_SIGN in drone/config.py")

    # --------------------------------------------------------------- go mode
    def move_to(self, target, speed):
        while True:
            self._check_abort()
            self.correct_yaw()
            x, y, z, _ = self.pose.get()
            dx, dy, dz = target[0] - x, target[1] - y, target[2] - z
            # The Tello can't do moves where every axis is < 20 cm. The remaining
            # error is not lost: the pose keeps the real position, so the next
            # move (or precise positioning) corrects for it.
            bx, by = self.pose.world_to_body(dx, dy)
            if max(abs(bx), abs(by), abs(dz)) < cfg.MIN_MOVE_CM:
                return
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            scale = min(1.0, cfg.MAX_STEP_CM / dist)
            bx, by = self.pose.world_to_body(dx * scale, dy * scale)
            bx, by, bz = int(round(bx)), int(round(by)), int(round(dz * scale))
            wdx, wdy = self.pose.body_to_world(bx, by)
            self.pose.begin_move((x + wdx, y + wdy, z + bz), speed)
            try:
                self.tello.go_xyz_speed(bx, by, bz, speed)
            except Exception:
                self.pose.cancel_move()
                raise
            self.pose.end_move()

    # --------------------------------------------------------------- rc steering
    def send_velocity(self, vx, vy, vz, min_units=0):
        """World velocity (cm/s) -> rc command in the body frame, with heading hold."""
        bx, by = self.pose.world_to_body(vx, vy)
        fb, lr, ud = bx / cfg.RC_CMS_PER_UNIT, -by / cfg.RC_CMS_PER_UNIT, vz / cfg.RC_CMS_PER_UNIT
        mag = math.sqrt(fb * fb + lr * lr + ud * ud)
        if 0 < mag < min_units:  # the Tello ignores very small rc values
            fb, lr, ud = (v * min_units / mag for v in (fb, lr, ud))
        yv = -cfg.RC_YAW_GAIN * self.yaw_error() if self.yaw_hold else 0  # rc yaw: clockwise +
        clamp = lambda v: int(max(-100, min(100, round(v))))
        self.tello.send_rc_control(clamp(lr), clamp(fb), clamp(ud), clamp(yv))

    def rc_leg(self, start, target, speed, stop_at_end):
        """Follow the straight line start -> target without stopping (pure pursuit)."""
        ab = [t - s for s, t in zip(start, target)]
        length = math.sqrt(sum(c * c for c in ab))
        deadline = time.time() + length / speed * 3 + 6
        x, y, z, _ = self.pose.get()
        worst = math.dist((x, y, z), target) + 40  # safety: we must get closer, not further
        period = 1.0 / cfg.RC_HZ
        while True:
            self._check_abort()
            if not self.pose.measured:
                self.tello.send_rc_control(0, 0, 0, 0)
                raise OdometryLost()
            if time.time() > deadline:
                self.tello.send_rc_control(0, 0, 0, 0)
                raise OdometryLost()  # not making progress: use 'go' for the rest
            x, y, z, _ = self.pose.get()
            p = (x, y, z)
            to_target = math.dist(p, target)
            if to_target < (cfg.FINE_TOL_CM if stop_at_end else cfg.RC_PASS_CM):
                break
            if to_target > worst:
                self.tello.send_rc_control(0, 0, 0, 0)
                self.pose.vo_disabled = True
                print("⚠️  Drone gaat van het punt weg: vloeiend vliegen gestopt en "
                      "odometrie uitgeschakeld")
                raise OdometryLost()
            # aim point: closest point on the path + lookahead
            if length > 0:
                along = sum((pc - sc) * c for pc, sc, c in zip(p, start, ab)) / length
                s = min(max(along, 0.0) + cfg.RC_LOOKAHEAD_CM, length)
                aim = [sc + c * s / length for sc, c in zip(start, ab)]
            else:
                aim = list(target)
            err = [a - pc for a, pc in zip(aim, p)]
            n = math.sqrt(sum(e * e for e in err)) or 1.0
            v = speed
            if stop_at_end:  # slow down towards the last point
                v = min(v, cfg.RC_GAIN * to_target + cfg.RC_MIN_UNITS * cfg.RC_CMS_PER_UNIT)
            self.send_velocity(*(e / n * v for e in err), min_units=cfg.RC_MIN_UNITS)
            time.sleep(period)
        if stop_at_end:
            self.tello.send_rc_control(0, 0, 0, 0)

    def hold_position(self, target):
        """Precise positioning on a waypoint with small rc corrections."""
        if not self.pose.measured:
            return
        deadline = time.time() + cfg.FINE_TIMEOUT_S
        period = 1.0 / cfg.RC_HZ
        max_speed = 20.0
        ex = ey = 0.0
        try:
            while time.time() < deadline:
                self._check_abort()
                if not self.pose.measured:
                    return
                x, y, z, _ = self.pose.get()
                ex, ey, ez = target[0] - x, target[1] - y, target[2] - z
                if math.hypot(ex, ey) < cfg.FINE_TOL_CM and abs(ez) < 1.5 * cfg.FINE_TOL_CM:
                    return
                v = [cfg.RC_GAIN * e for e in (ex, ey, ez)]
                n = math.sqrt(sum(c * c for c in v))
                if n > max_speed:
                    v = [c * max_speed / n for c in v]
                self.send_velocity(*v, min_units=cfg.RC_MIN_UNITS)
                time.sleep(period)
            print(f"ℹ️  Nauwkeurig positioneren: na {cfg.FINE_TIMEOUT_S}s nog "
                  f"{math.hypot(ex, ey):.0f} cm naast het punt")
        finally:
            self.tello.send_rc_control(0, 0, 0, 0)
