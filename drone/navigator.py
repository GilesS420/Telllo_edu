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


class FlightInterrupted(Exception):
    """Yield navigation to a priority flip without landing."""


class OdometryLost(Exception):
    pass


class MotionEstimate:
    """Filtered measured velocity; repeated camera samples carry no new speed."""

    def __init__(self):
        self.previous = None
        self.velocity = (0.0, 0.0, 0.0)

    def observe(self, position, timestamp):
        if self.previous is not None:
            previous_time, previous_position = self.previous
            dt = timestamp - previous_time
            if dt <= 0:
                return self.velocity
            if dt < 0.7:
                alpha = dt / (0.2 + dt)
                raw = tuple(max(-150, min(150, (new - old) / dt))
                            for new, old in zip(position, previous_position))
                self.velocity = tuple(old + alpha * (new - old)
                                      for old, new in zip(self.velocity, raw))
            else:
                self.velocity = (0.0, 0.0, 0.0)
        self.previous = (timestamp, tuple(position))
        return self.velocity


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
        self.position_valid = True  # continuity with the flight origin is known
        self.local_reference_valid = False  # measured continuity with a saved click, not the origin
        self.measurement_time = None
        self.quality = 0.0      # last phase correlation response (for the GUI)
        self.vo_disabled = False  # set when the odometry contradicts the commands
        self.vo_validated = False  # odometry matched at least one commanded move
        self.cam_turn_checked = False  # camera rotation auto-correction used

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

    def reset(self):
        """
        New mission frame from the drone itself: origin = where it is now, x = the
        direction its nose (front camera) points now, y = left. Only on the ground.
        """
        with self._lock:
            self.x = self.y = 0.0
            self.yaw = self.reference_yaw = 0.0
            self._yaw_offset = None   # re-anchored at the next IMU reading
            self._seg = None
            self._hist.clear()
            self.measured = False
            self.position_valid = True
            self.local_reference_valid = False
            self.measurement_time = None
            self.quality = 0.0
            # a new flight gets a new chance: a wrong measurement in an earlier
            # flight (e.g. a dark floor) must not switch the odometry off for good
            self.vo_disabled = False
            self.vo_validated = False

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
                self.position_valid = False  # commanded estimate, not a measured arrival
                self.local_reference_valid = False
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
        # The measured move points the wrong way. If it has the right length and is
        # turned by 90/180/270 deg, the camera image is simply mounted turned:
        # correct that once (first check of the flight) instead of giving up.
        m = math.hypot(mx, my)
        turn = math.degrees(math.atan2(cy, cx) - math.atan2(my, mx))
        quarter = round(turn / 90) * 90
        if (not self.vo_validated and not self.cam_turn_checked and 0.4 * c < m < 2.5 * c
                and abs(wrap_deg(turn - quarter)) < 25 and quarter % 360):
            self.cam_turn_checked = True
            cfg.CAM_ROTATE_DEG = int((cfg.CAM_ROTATE_DEG + quarter) % 360)
            self.vo_validated = True
            self.x, self.y = seg["end"][0], seg["end"][1]
            print(f"🔄 Camerabeeld is {wrap_deg(quarter):+.0f}° gedraaid t.o.v. de drone: "
                  f"gecorrigeerd. Zet CAM_ROTATE_DEG = {cfg.CAM_ROTATE_DEG} in drone/config.py")
            return
        # Never trust it again this flight: a wrong camera direction would make
        # every correction push the drone further away.
        self.vo_disabled = True
        self.measured = False
        self.position_valid = False
        self.local_reference_valid = False
        self.x, self.y = seg["end"][0], seg["end"][1]
        print(f"⚠️  Odometrie mat {along:.0f} cm voor een beweging van {c:.0f} cm: odometrie "
              "UITGESCHAKELD. Controleer CAM_ROTATE_DEG / CAM_FORWARD_SIGN / CAM_LEFT_SIGN / "
              "CAM_HFOV_DEG.")

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
                self.measurement_time = now
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

    def snapshot(self):
        """Atomically capture coordinates and their tracking validity at a click."""
        with self._lock:
            return {"pose": (self.x, self.y, self.z, self.yaw),
                    "tracking_valid": self.measured and self.position_valid,
                    "measured": self.measured, "position_valid": self.position_valid,
                    "local_reference_valid": self.local_reference_valid,
                    "measurement_time": self.measurement_time,
                    "quality": self.quality}

    def relocalize_xy(self, dx, dy):
        """Restore the original frame from a measured known landmark, while idle."""
        with self._lock:
            if self._seg is not None or not self.measured or not self.local_reference_valid:
                return False
            self.x += dx
            self.y += dy
            self.position_valid = True
            self._hist.clear()  # delayed video must not reuse poses from the old offset
            return True

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
        self._last_rc = 0.0
        self.level = False   # True: never correct the height (fly level, ignore the floor)
        self.interruptible = False
        self.flip_pending = threading.Event()
        if cfg.MAX_STEP_CM < 40 or cfg.MAX_STEP_CM > 500:
            raise ValueError("MAX_STEP_CM must be between 40 and 500")

    def rc(self, lr, fb, ud, yv):
        """send_rc_control that remembers when the sticks were last used."""
        if any((lr, fb, ud, yv)):
            self._last_rc = time.time()
        self.tello.send_rc_control(lr, fb, ud, yv)

    def settle_before_flip(self):
        """Wait only for the remaining RC transition; never retry a flip."""
        if self._last_rc <= 0:
            return 0.0
        remaining = cfg.RC_SETTLE_S - (time.time() - self._last_rc)
        if remaining <= 0:
            return 0.0
        started = time.time()
        self.rc(0, 0, 0, 0)
        while remaining > 0:
            self._check_abort()
            time.sleep(min(0.02, remaining))
            remaining = cfg.RC_SETTLE_S - (time.time() - self._last_rc)
        return max(0.0, time.time() - started)

    def command(self, fn, *args):
        """
        Send an SDK command (go, cw/ccw, land, ...). The Tello answers 'error Not
        joystick' when such a command arrives while it is still busy with rc (stick)
        commands, e.g. right after the recentering at takeoff or precise
        positioning: wait until it has settled, and retry.
        """
        wait = cfg.RC_SETTLE_S - (time.time() - self._last_rc)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                return fn(*args)
            except Exception as e:
                if "joystick" not in str(e).lower() or attempt == 2:
                    raise
                print("ℹ️  Tello nog bezig met de vorige beweging: even wachten en opnieuw")
                self.tello.send_rc_control(0, 0, 0, 0)
                time.sleep(1.5)

    def _dz(self, dz):
        """Height error to correct: none within Z_DEADBAND_CM, none at all when flying level."""
        return 0.0 if self.level or abs(dz) < cfg.Z_DEADBAND_CM else dz

    def _check_abort(self):
        if self.abort.is_set():
            raise MissionAborted()
        if self.interruptible and self.flip_pending.is_set():
            raise FlightInterrupted()

    # ------------------------------------------------------------------ mission
    def fly(self, waypoints, speed, on_waypoint=None, mode=None, fine=None):
        """Fly through all waypoints. Raises MissionAborted when abort is set."""
        speed = int(min(max(speed, 10), 100))
        mode = mode or cfg.NAV_MODE
        fine = fine or cfg.FINE_POSITION
        if fine == "auto":
            fine = "last"
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
                        self.rc(0, 0, 0, 0)
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
                self.rc(0, 0, 0, 0)

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
            self.command(self.tello.rotate_counter_clockwise, deg)
        else:
            self.command(self.tello.rotate_clockwise, deg)
        time.sleep(0.3)  # let the IMU reading catch up
        new_err = self.yaw_error()
        if abs(new_err) > abs(err) + 2:
            # Rotating made it worse: the yaw sign is wrong. Never keep spinning.
            self.yaw_hold = False
            print("⚠️  Richting werd slechter na correctie: richting vasthouden UIT. "
                  "Controleer YAW_SIGN in drone/config.py")

    # --------------------------------------------------------------- go mode
    def move_to(self, target, speed, require_tracking=False):
        while True:
            self._check_abort()
            if require_tracking and not (self.pose.measured and self.pose.position_valid):
                raise OdometryLost("Global return requires continuous measured position")
            self.correct_yaw()
            x, y, z, _ = self.pose.get()
            dx, dy = target[0] - x, target[1] - y
            dz = self._dz(target[2] - z)   # small floor step: keep flying level
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
                self.command(self.tello.go_xyz_speed, bx, by, bz, speed)
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
        self.rc(clamp(lr), clamp(fb), clamp(ud), clamp(yv))

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
                self.rc(0, 0, 0, 0)
                raise OdometryLost()
            if time.time() > deadline:
                self.rc(0, 0, 0, 0)
                raise OdometryLost()  # not making progress: use 'go' for the rest
            x, y, z, _ = self.pose.get()
            if not self._dz(target[2] - z):
                z = target[2]    # small floor step: fly level, don't chase the ToF height
            p = (x, y, z)
            to_target = math.dist(p, target)
            if to_target < (cfg.FINE_TOL_CM if stop_at_end else cfg.RC_PASS_CM):
                break
            if to_target > worst:
                self.rc(0, 0, 0, 0)
                self.pose.vo_disabled = True
                print("⚠️  Drone gaat van het punt weg: vloeiend vliegen gestopt en "
                      "odometrie uitgeschakeld")
                raise OdometryLost()
            # aim point: closest point on the path + lookahead. Within PATH_TOL_CM
            # beside the path the drone flies parallel to it instead of steering
            # back: every small correction bends the path, which looks worse
            # than a steady few cm offset.
            if length > 0:
                along = sum((pc - sc) * c for pc, sc, c in zip(p, start, ab)) / length
                s = min(max(along, 0.0) + cfg.RC_LOOKAHEAD_CM, length)
                aim = [sc + c * s / length for sc, c in zip(start, ab)]
                a = min(max(along, 0.0), length)
                cross = [pc - (sc + c * a / length) for pc, sc, c in zip(p, start, ab)]
                off = math.sqrt(sum(c * c for c in cross))
                if off > 0 and to_target > cfg.PATH_TOL_CM:
                    keep = min(off, cfg.PATH_TOL_CM) / off   # part of the offset we accept
                    aim = [am + c * keep for am, c in zip(aim, cross)]
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
            self.rc(0, 0, 0, 0)

    def descend_and_land(self, target_xy):
        """
        Land on target_xy: descend slowly with rc while the odometry keeps the
        drone above the spot, then the normal 'land' for the last few cm. The
        Tello's own landing from 80 cm slides sideways much more.
        Returns False (nothing sent) when the odometry isn't tracking.
        """
        if not self.pose.measured or not self.pose.position_valid:
            return False
        self.hold_position((*target_xy, self.pose.get()[2]))
        period = 1.0 / cfg.RC_HZ
        _, _, z, _ = self.pose.get()
        deadline = time.time() + max(z - cfg.LAND_HOVER_CM, 0) / cfg.LAND_DESCENT_CMS + 4
        try:
            while time.time() < deadline:
                self._check_abort()
                x, y, z, _ = self.pose.get()
                if z <= cfg.LAND_HOVER_CM or not self.pose.measured or not self.pose.position_valid:
                    break
                vx, vy = (cfg.RC_GAIN * (t - c) for t, c in zip(target_xy, (x, y)))
                n = math.hypot(vx, vy)
                if n > 20:
                    vx, vy = vx * 20 / n, vy * 20 / n
                self.send_velocity(vx, vy, -cfg.LAND_DESCENT_CMS)
                time.sleep(period)
        finally:
            self.rc(0, 0, 0, 0)
        return True

    def hold_position(self, target, max_speed=20.0, gain=None,
                      tolerance_cm=None, timeout_s=None, settle_s=0.0,
                      allow_local_reference=False, damping=0.0, max_vertical_speed=None,
                      max_error_growth_cm=15):
        """Precise positioning on a waypoint with small rc corrections."""
        def reference_valid():
            return self.pose.position_valid or (allow_local_reference and self.pose.local_reference_valid)

        self.last_hold_result = {"target_cm": tuple(target), "reason": "running", "trace": []}
        def finish(result, reason):
            self.last_hold_result.update(confirmed=result, reason=reason)
            return result

        if not self.pose.measured or not reference_valid():
            return finish(False, "tracking_unavailable")
        t0 = time.time()
        timeout_s = cfg.FINE_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = t0 + timeout_s
        tolerance = cfg.FINE_TOL_CM if tolerance_cm is None else tolerance_cm
        z_tolerance = 1.5 * tolerance if tolerance_cm is None else tolerance
        # A noisy derivative must not reject a position that stays inside the
        # band for the full settling period. Still reject fast crossings.
        settle_speed = max(3.0, min(6.0, tolerance / max(settle_s, 0.5)))
        stable_since = None
        period = 1.0 / cfg.RC_HZ
        gain = cfg.RC_GAIN if gain is None else gain
        self.last_hold_result.update(gain=gain, damping=damping, timeout_s=timeout_s,
                                     max_speed=max_speed, max_vertical_speed=max_vertical_speed)
        ex = ey = 0.0
        e0 = None
        motion = MotionEstimate()
        try:
            while time.time() < deadline:
                self._check_abort()
                if not reference_valid():
                    return finish(False, "reference_lost")
                if not self.pose.measured:
                    # Acceleration/brief blur can interrupt floor tracking.
                    # Hover until it returns within the existing deadline;
                    # never continue steering on an old error or claim success.
                    stable_since = None
                    self.rc(0, 0, 0, 0)
                    time.sleep(period)
                    continue
                snapshot = self.pose.snapshot()
                x, y, z, _ = snapshot["pose"]
                sample_time = snapshot["measurement_time"]
                now = time.time()
                if sample_time is not None and now - sample_time > 0.35:
                    stable_since = None
                    self.rc(0, 0, 0, 0)
                    time.sleep(period)
                    continue
                velocity = motion.observe((x, y, z), now if sample_time is None else sample_time)
                e = math.hypot(target[0] - x, target[1] - y)
                e0 = e if e0 is None else e0
                # safety: the error must shrink. If not, the camera direction is
                # wrong (not yet checked this flight): stop instead of wandering off.
                if e > e0 + max_error_growth_cm or (time.time() - t0 > 2.5 and e > 0.9 * e0 + 5):
                    print("⚠️  Nauwkeurig positioneren komt niet dichter: gestopt")
                    return finish(False, "error_not_decreasing_check_camera_calibration")
                # Explicit precision also corrects small altitude errors; the
                # normal mission deadband must not erase a flip's height error.
                dz = target[2] - z
                ex, ey, ez = target[0] - x, target[1] - y, (dz if tolerance_cm is not None else self._dz(dz))
                xy_slow = not damping or math.hypot(velocity[0], velocity[1]) < settle_speed
                z_slow = not damping or abs(velocity[2]) < settle_speed
                if math.hypot(ex, ey) < tolerance and abs(ez) < z_tolerance and xy_slow and z_slow:
                    stable_since = time.time() if stable_since is None else stable_since
                    if time.time() - stable_since >= settle_s:
                        return finish(True, "stable_within_tolerance")
                    self.rc(0, 0, 0, 0)
                    time.sleep(period)
                    continue
                stable_since = None
                v = [gain * ex - damping * velocity[0], gain * ey - damping * velocity[1],
                     (min(gain, 1.0) if damping else gain) * ez - damping * velocity[2]]
                # A fast pass through the tolerance band is not an arrival.
                # Brake residual measured motion instead of coasting past the
                # target, then leave a settled axis alone.
                if math.hypot(ex, ey) < tolerance:
                    v[0], v[1] = ((0.0, 0.0) if xy_slow else
                                  (-damping * velocity[0], -damping * velocity[1]))
                if abs(ez) < z_tolerance:
                    v[2] = 0.0 if z_slow else -damping * velocity[2]
                if max_vertical_speed is not None:
                    v[2] = max(-max_vertical_speed, min(max_vertical_speed, v[2]))
                n = math.sqrt(sum(c * c for c in v))
                if n > max_speed:
                    v = [c * max_speed / n for c in v]
                self.last_hold_result["trace"].append(
                    [round(time.time() - t0, 3), round(x, 2), round(y, 2), round(z, 2),
                     *(round(c, 2) for c in v)])
                # Outside tolerance even a fine correction must overcome the
                # drone's stick dead zone. Stop inside tolerance, rather than
                # spending seconds sending tiny ineffective commands.
                self.send_velocity(*v, min_units=cfg.RC_MIN_UNITS)
                time.sleep(period)
            print(f"ℹ️  Nauwkeurig positioneren: na {timeout_s}s nog "
                  f"{math.hypot(ex, ey):.0f} cm naast het punt")
            return finish(False, "timeout")
        finally:
            self.rc(0, 0, 0, 0)
