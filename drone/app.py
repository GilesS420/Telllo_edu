"""
DroneApp: the core of the autonomous Tello. Flies missions (from the GUI or the
Jetson) and reports puddles seen by the downward camera. Started by tello_gui.py.

Threads:
    mission   the ONLY thread that sends flight commands to the Tello
    detector  puddle detection on the downward camera
    odometry  heading, height and visual odometry -> pose (odometry.py)
    telemetry all sensor values + terrain map, for the GUI (telemetry.py)
    status    status messages to the Jetson
    link      receives Jetson messages (jetson_link.py)
"""

import math
import os
import queue
import threading
import time

import cv2

from . import config as cfg
from .helipad import HelipadDetector, draw_helipad
from .jetson_link import JetsonLink
from .navigator import MissionAborted, PathFollower, Pose, validate_waypoints
from .odometry import Odometry
from .puddle_detector import PuddleTracker, create_detector, draw_detections, pixel_to_world
from .telemetry import Telemetry


class DroneApp:
    def __init__(self, tello, record_dir=None):
        self.tello = tello
        self.pose = Pose()
        self.abort = threading.Event()
        self.running = True
        self.state = "idle"          # idle, taking_off, flying, landing, error
        self.airborne = False
        self.mission_id = None
        self.current_waypoints = []  # waypoints of the running/last mission (for the GUI)
        self.waypoint_index = -1
        self.waypoints_reached = 0   # reached waypoints of the current mission (for the GUI)
        self.commands = queue.Queue()
        self.follower = PathFollower(tello, self.pose, self.abort)
        self.detector = create_detector()
        self.tracker = PuddleTracker()
        self.pad_detector = HelipadDetector()
        self.pads = []               # H sightings this mission: (t, x, y, size), newest last
        self._frame = None
        self._vis = None
        self._vis_lock = threading.Lock()
        self._last_cmd_time = time.time()
        self.record_dir = record_dir or cfg.RECORD_DIR
        self.recording = record_dir is not None
        self._last_record = 0.0
        self._record_count = 0
        self.downvision_enabled = False

        self.link = JetsonLink(self.submit, cfg.LISTEN_HOST, cfg.LISTEN_PORT,
                               cfg.JETSON_HOST, cfg.JETSON_PORT)

    # ------------------------------------------------------------------ setup
    def start(self):
        self.tello.connect()
        print("Battery level:", self.tello.get_battery())
        self.tello.streamon()
        self.set_downvision(True)
        self.frame_reader = self.tello.get_frame_read()
        self.mission_thread = threading.Thread(target=self.mission_loop, daemon=True)
        self.mission_thread.start()
        threading.Thread(target=self.detect_loop, daemon=True).start()
        self.odometry = Odometry(self.tello, self.pose, self.frame_reader, self.detector,
                                 lambda: self.airborne)
        self.telemetry = Telemetry(self.tello, self.pose, lambda: self.airborne)
        threading.Thread(target=self.status_loop, daemon=True).start()

    def set_downvision(self, enabled):
        enabled = bool(enabled)
        if self.downvision_enabled == enabled:
            return
        self.tello.send_command_with_return(f"downvision {1 if enabled else 0}")
        self.downvision_enabled = enabled
        print(f"Downvision {'enabled' if enabled else 'disabled'}")

    def toggle_downvision(self):
        self.set_downvision(not self.downvision_enabled)

    # ------------------------------------------------- jetson / manual input
    def submit(self, msg):
        """
        Entry point for every command, from the GUI or the Jetson.
        Runs in the caller's thread: never send flight commands from here.
        """
        t = msg["type"]
        if t in ("abort", "land"):
            if t == "abort":
                print("🛑 Abort")
            self.abort.set()  # stop a running mission after its current step
            self.commands.put({"type": "land"})
        elif t == "ping":
            self.link.send({"type": "pong", "t": msg.get("t")})
        elif t == "get_puddles":
            self.link.send({"type": "puddle_list", "puddles": self.tracker.confirmed()})
        elif t == "reset_puddles":
            self.tracker.reset()
            self.link.send({"type": "ack", "ref": t})
        elif t in ("mission", "takeoff", "set_pose"):
            self.commands.put(msg)
        else:
            self.link.send({"type": "error", "ref": t, "error": "unknown message type"})

    def send_status(self):
        x, y, z, yaw = self.pose.get()
        try:
            battery = self.tello.get_battery()
        except Exception:
            battery = None
        self.link.send({
            "type": "status", "state": self.state, "battery": battery,
            "pos": {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1), "yaw": round(yaw, 1)},
            "odometry": self.pose.measured,
            "mission": self.mission_id, "waypoint_index": self.waypoint_index,
            "puddles": len(self.tracker.confirmed()),
        })

    def status_loop(self):
        while self.running:
            self.send_status()
            time.sleep(cfg.STATUS_INTERVAL)

    # --------------------------------------------------------- mission thread
    def mission_loop(self):
        while self.running:
            try:
                cmd = self.commands.get(timeout=0.5)
            except queue.Empty:
                # Tello lands automatically after 15 s without commands
                if self.airborne and time.time() - self._last_cmd_time > cfg.KEEPALIVE_INTERVAL:
                    self.tello.send_rc_control(0, 0, 0, 0)
                    self._last_cmd_time = time.time()
                continue
            try:
                self.handle_command(cmd)
            except Exception as e:
                print(f"❌ {cmd.get('type')} failed: {e}")
                self.state = "error"
                self.link.send({"type": "error", "ref": cmd.get("type"), "error": str(e)})
                self.safe_land()
            self._last_cmd_time = time.time()

    def handle_command(self, cmd):
        t = cmd["type"]
        if t == "mission":
            self.run_mission(cmd)
        elif t == "land":
            self.safe_land()
        elif t == "takeoff":
            self.takeoff()
        elif t == "set_pose":
            if self.airborne:
                raise RuntimeError("set_pose is only allowed on the ground")
            self.pose.set(cmd.get("x"), cmd.get("y"), None, cmd.get("yaw"))
            self.link.send({"type": "ack", "ref": t})

    def takeoff(self):
        if self.airborne:
            return
        battery = self.tello.get_battery()
        if battery < cfg.MIN_BATTERY:
            raise RuntimeError(f"battery too low ({battery}% < {cfg.MIN_BATTERY}%)")
        self.state = "taking_off"
        print("🚁 Opstijgen...")
        x0, y0, _, _ = self.pose.get()
        self.airborne = True  # odometry measures the drift during takeoff too
        try:
            self.tello.takeoff()
        except Exception:
            self.airborne = False
            raise
        height = self.read_height(default=80)
        self.pose.set(z=height)
        self.state = "flying"
        x, y, _, _ = self.pose.get()
        print(f"✅ In de lucht op {height} cm ({x - x0:+.0f}, {y - y0:+.0f} cm verschoven)")
        if cfg.TAKEOFF_RECENTER and self.pose.measured and math.hypot(x - x0, y - y0) > 3:
            print("↩️  Terug boven de startplek")
            self.follower.hold_position((x0, y0, height))

    def safe_land(self, spot=None):
        """Land. With spot=(x, y) and working odometry: precise landing on that spot."""
        if not self.airborne:
            self.state = "idle"
            return
        self.state = "landing"
        print("🛬 Landen...")
        if spot is not None and cfg.HELIPAD_LAND:
            try:
                if self.land_on_helipad(spot):
                    spot = None
            except MissionAborted:
                spot = None
            except Exception as e:  # never stay in the air because of this
                print(f"⚠️  Landen op de H mislukt ({e}), gewoon landen")
        if spot is not None and cfg.PRECISE_LAND:
            try:
                if self.follower.descend_and_land(spot):
                    print("🎯 Precies boven het landingspunt gedaald")
            except Exception as e:  # never stay in the air because of this
                print(f"⚠️  Precies landen mislukt ({e}), gewoon landen")
        try:
            self.tello.land()
        except Exception as e:
            print(f"⚠️  Land command failed: {e}")
        self.airborne = False
        self.pose.set(z=0)
        self.state = "idle"

    def run_mission(self, msg):
        waypoints = validate_waypoints(msg["waypoints"])
        speed = msg.get("speed", cfg.DEFAULT_SPEED)
        land_at_end = msg.get("land_at_end", cfg.LAND_AT_END)
        self.mission_id = msg.get("id")
        self.current_waypoints = waypoints
        self.waypoints_reached = 0
        self.abort.clear()
        self.pads = []
        print(f"🗺️  Mission {self.mission_id}: {len(waypoints)} waypoints @ {speed} cm/s")
        self.link.send({"type": "ack", "ref": "mission", "id": self.mission_id,
                        "waypoints": len(waypoints)})
        self.takeoff()

        def on_waypoint(i, wp):
            self.waypoint_index = i
            self.waypoints_reached = i + 1
            x, y, z, _ = self.pose.get()
            print(f"📍 Waypoint {i + 1}/{len(waypoints)} bereikt ({x:.0f}, {y:.0f}, {z:.0f})")
            self.link.send({"type": "waypoint_reached", "id": self.mission_id, "index": i,
                            "pos": {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1)}})

        try:
            self.follower.fly(waypoints, speed, on_waypoint,
                              mode=msg.get("nav_mode"), fine=msg.get("fine"))
        except MissionAborted:
            print("🛑 Mission aborted")
            self.link.send({"type": "mission_aborted", "id": self.mission_id,
                            "puddles": self.tracker.confirmed()})
            self.safe_land()
            return
        finally:
            self.waypoint_index = -1

        print("🏁 Mission complete")
        self.link.send({"type": "mission_done", "id": self.mission_id,
                        "puddles": self.tracker.confirmed()})
        if land_at_end:
            self.safe_land(spot=waypoints[-1][:2])

    def _pad_near(self, xy, since):
        """Newest H sighting after time `since` within HELIPAD_RADIUS_CM of xy, or None."""
        for t, x, y, size in reversed(self.pads):
            if t < since:
                break
            if math.hypot(x - xy[0], y - xy[1]) <= cfg.HELIPAD_RADIUS_CM:
                return t, x, y, size
        return None

    def _look_for_pad(self, end_xy, climb_cms):
        """Hover (or climb to HELIPAD_SEARCH_HEIGHT_CM) while looking for the H."""
        t_end = time.time() + cfg.HELIPAD_SEARCH_S
        if climb_cms:
            t_end += 6
        try:
            while time.time() < t_end:
                self.follower._check_abort()
                found = self._pad_near(end_xy, 0)
                if found:
                    return found
                x, y, z, _ = self.pose.get()
                vz = climb_cms if climb_cms and z < cfg.HELIPAD_SEARCH_HEIGHT_CM else 0
                # stay above the end point (otherwise the drone drifts away while looking)
                vx, vy = (cfg.RC_GAIN * (e - c) for e, c in zip(end_xy, (x, y)))
                n = math.hypot(vx, vy)
                if n > 15:
                    vx, vy = vx * 15 / n, vy * 15 / n
                self.follower.send_velocity(vx, vy, vz)
                time.sleep(0.1)
            return None
        finally:
            self.tello.send_rc_control(0, 0, 0, 0)

    def land_on_helipad(self, end_xy):
        """
        Look for an H near the end point, steer above its centre and descend.
        The H position is measured again in every camera frame, so the drone
        keeps correcting while it comes down. Returns False when there is no H
        (nothing done), True when the drone is low above the H (or gave up
        there) and only the final 'land' is left.
        """
        f = self.follower
        sighting = self._pad_near(end_xy, 0)
        if sighting is None:   # not seen on the way: hover at the end point and look
            sighting = self._look_for_pad(end_xy, 0.0)
        if sighting is None and self.pose.get()[2] < cfg.HELIPAD_SEARCH_HEIGHT_CM - 15:
            # higher up the camera sees a bigger part of the floor
            print(f"🔎 Geen H in beeld: stijgen naar {cfg.HELIPAD_SEARCH_HEIGHT_CM} cm om te zoeken")
            sighting = self._look_for_pad(end_xy, cfg.HELIPAD_CLIMB_CMS)
        if sighting is None:
            print("ℹ️  Geen H gevonden bij het eindpunt: landen op de coördinaten")
            return False
        t_seen, tx, ty, size = sighting
        print(f"🅷 H gevonden op ({tx:.0f}, {ty:.0f}): erboven centreren en dalen")
        period = 1.0 / cfg.RC_HZ
        deadline = time.time() + cfg.HELIPAD_TIMEOUT_S
        try:
            while time.time() < deadline:
                f._check_abort()
                new = self._pad_near(end_xy, t_seen + 1e-6)
                if new:   # fresh measurement of the H (smoothed a little)
                    t_seen, size = new[0], new[3]
                    tx, ty = 0.4 * tx + 0.6 * new[1], 0.4 * ty + 0.6 * new[2]
                x, y, z, _ = self.pose.get()
                ex, ey = tx - x, ty - y
                err = math.hypot(ex, ey)
                lost = time.time() - t_seen
                if lost > 3.0 and not self.pose.measured:
                    print("⚠️  H en odometrie kwijt: hier landen")
                    return True
                # low enough, or the H almost fills the image (lower it would
                # get cut off by the image edge and can't be seen any more)
                big = size >= cfg.HELIPAD_FINAL_SIZE and lost < 1.0
                if err < cfg.HELIPAD_CENTER_TOL_CM and (z <= cfg.HELIPAD_FINAL_CM or big):
                    print(f"🎯 Boven het midden van de H ({err:.0f} cm ernaast)")
                    return True
                if lost > 5.0 and err < cfg.FINE_TOL_CM:
                    print("ℹ️  H niet meer te zien: landen op de laatst gemeten plek")
                    return True
                # only come down while centred and the H is in view; otherwise
                # hold the height and steer back above it
                centred = err < cfg.HELIPAD_CENTER_TOL_CM * (2 if z > 60 else 1)
                vz = -cfg.HELIPAD_DESCENT_CMS if centred and lost < 1.0 else 0.0
                vx, vy = cfg.RC_GAIN * ex, cfg.RC_GAIN * ey
                n = math.hypot(vx, vy)
                if n > 15:
                    vx, vy = vx * 15 / n, vy * 15 / n
                f.send_velocity(vx, vy, vz, min_units=0 if centred else cfg.RC_MIN_UNITS)
                time.sleep(period)
            print(f"ℹ️  H: na {cfg.HELIPAD_TIMEOUT_S}s nog niet gecentreerd, hier landen")
            return True
        finally:
            self.tello.send_rc_control(0, 0, 0, 0)

    def read_height(self, default):
        """Height above the floor from the ToF sensor (cm), fallback to default."""
        try:
            h = self.tello.get_distance_tof()
            if 10 < h < 1000:
                return h
        except Exception:
            pass
        return default

    # -------------------------------------------------------- detector thread
    def detect_loop(self):
        period = 1.0 / cfg.DETECT_HZ
        while self.running:
            t_start = time.time()
            frame = self.frame_reader.frame
            if frame is not None:
                self.process_frame(frame, t_start)
            time.sleep(max(0.0, period - (time.time() - t_start)))

    def emergency_stop(self):
        """Motors off immediately. The drone falls!"""
        print("🚨 EMERGENCY STOP")
        self.abort.set()
        self.tello.emergency()
        self.airborne = False
        self.state = "idle"

    def toggle_recording(self):
        self.recording = not self.recording
        print(f"🎥 Opnemen {'AAN -> ' + self.record_dir if self.recording else 'UIT'}")

    def record_frame(self, gray):
        """Save downward camera frames as training images for Roboflow."""
        now = time.time()
        if now - self._last_record < 1.0 / cfg.RECORD_HZ:
            return
        if cfg.RECORD_ONLY_AIRBORNE and not self.airborne:
            return
        self._last_record = now
        os.makedirs(self.record_dir, exist_ok=True)
        path = os.path.join(self.record_dir, f"down_{now:.2f}.png")
        cv2.imwrite(path, gray)
        self._record_count += 1

    def latest_vis(self):
        """Last processed downward image with the detections drawn (BGR), or None."""
        with self._vis_lock:
            return self._vis

    def process_frame(self, frame, t_frame):
        pose = self.pose.get(t_frame - cfg.FRAME_LATENCY_S)
        dets, gray, _ = self.detector.detect(frame)
        if self.recording:
            self.record_frame(gray)
        height = self.read_height(default=pose[2])
        active = self.airborne and self.state == "flying" and height >= cfg.MIN_DETECT_HEIGHT
        pad = None
        if cfg.HELIPAD_LAND and self.airborne and height >= 15:
            pad = self.pad_detector.find(gray)
            if pad is not None:
                h, w = gray.shape
                x, y, _ = pixel_to_world(pad.cx, pad.cy, w, h, height, pose)
                self.pads.append((t_frame, x, y, pad.size_px / w))
                del self.pads[:-50]
        if active:
            h, w = gray.shape
            for d in dets:
                if pad is not None and math.hypot(d.cx - pad.cx, d.cy - pad.cy) < pad.size_px:
                    continue  # the H itself is not a puddle
                x, y, s = pixel_to_world(d.cx, d.cy, w, h, height, pose)
                report = self.tracker.add(x, y, d.area_px * s * s)
                if report:
                    print(f"💧 Plas #{report['id']} op x={report['x']} y={report['y']} "
                          f"({report['area_cm2']:.0f} cm²)")
                    self.link.send({"type": "puddle", "mission": self.mission_id, **report})
        vis = draw_helipad(draw_detections(gray, dets if active else []), pad)
        with self._vis_lock:
            self._frame = frame.copy() if hasattr(frame, "copy") else frame
            self._vis = vis

    def shutdown(self):
        print("Cleaning up...")
        self.abort.set()
        self.running = False
        if hasattr(self, "telemetry"):
            self.telemetry.stop()
        # Wait until the mission thread finished its current move, so only one
        # thread talks to the Tello at a time.
        self.mission_thread.join(timeout=cfg.RESPONSE_TIMEOUT)
        self.safe_land()
        try:
            self.tello.streamoff()
        except Exception:
            pass
        self.link.close()
        print("Drone landed and disconnected")
