"""
DroneApp: the core of the autonomous Tello. Flies missions (from the GUI or the
Jetson) and can land on an H landing pad seen by the downward camera. Started by
tello_gui.py. Object detection (puddles etc.) is done by a trained model
elsewhere, not in this code.

Threads:
    mission   the ONLY thread that sends flight commands to the Tello
    camera    downward camera: H landing pad detection, recording, GUI view
    odometry  heading, height and visual odometry -> pose (odometry.py)
    telemetry all sensor values + terrain map, for the GUI (telemetry.py)
    status    status messages to the Jetson
    link      receives Jetson messages (jetson_link.py)
"""

import json
import math
import os
import queue
import sys
import threading
import time
import traceback

import cv2

from . import config as cfg
from .downcam import DownCam, pixel_to_world
from .helipad import HelipadDetector, HelipadTracker, draw_helipad
from .jetson_link import JetsonLink
from .navigator import FlightInterrupted, MissionAborted, MotionEstimate, OdometryLost, PathFollower, Pose, validate_waypoints
from .odometry import Odometry
from .telemetry import Telemetry


class CommandRejected(Exception):
    """Invalid command: reported back, but the drone keeps doing what it did."""


class FlightCommandQueue(queue.Queue):
    """Land first, then flips, then other commands; FIFO within each group."""

    def __init__(self):
        super().__init__()
        self.flip_pending = threading.Event()
        self.dispatching = threading.Event()

    def _put(self, item):
        priority = {"land": 0, "flip": 1}
        rank = priority.get(item["type"], 2)
        index = next((i for i, cmd in enumerate(self.queue)
                      if priority.get(cmd["type"], 2) > rank), len(self.queue))
        self.queue.insert(index, item)
        if item["type"] == "flip":
            self.flip_pending.set()

    def _get(self):
        self.dispatching.set()  # covers the gap between dequeue and command execution
        item = self.queue.popleft()
        if not any(cmd["type"] == "flip" for cmd in self.queue):
            self.flip_pending.clear()
        return item


class DroneApp:
    def __init__(self, tello, record_dir=None):
        self.tello = tello
        self.pose = Pose()
        self.abort = threading.Event()
        self._command_lock = threading.RLock()
        self.running = True
        self.state = "idle"          # idle, taking_off, flying, landing, error
        self.airborne = False
        self.mission_id = None
        self.current_waypoints = []  # waypoints of the running/last mission (for the GUI)
        self.waypoint_index = -1
        self.waypoints_reached = 0   # reached waypoints of the current mission (for the GUI)
        self.last_flip_record = None
        self.commands = FlightCommandQueue()
        self._gen = 0                # +1 on every Land/Abort: older queued commands are void
        self.follower = PathFollower(tello, self.pose, self.abort)
        self.follower.flip_pending = self.commands.flip_pending
        self.cam = DownCam()
        self.pad_detector = HelipadDetector()
        self.pad_tracker = HelipadTracker()
        self.pads = []               # H sightings this mission: (t, x, y, size), newest last
        self.known_pad = None        # confirmed H, retained for the whole coordinate frame
        self.local_pads = []
        self.local_known_pad = None
        self.ever_seen_h = False
        self._h_relocalization = []
        self._active_command = None
        self._live_pad = None       # fresh camera-relative offset, usable without global xy
        self.frame_id = 0            # +1 on every new mission frame (the GUI clears its trail)
        self._pose_set = False       # set_pose was used: keep that frame at the next takeoff
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
        self.odometry = Odometry(self.tello, self.pose, self.frame_reader, self.cam,
                                 lambda: self.airborne, lambda: self.downvision_enabled)
        self.telemetry = Telemetry(self.tello, self.pose, lambda: self.airborne)
        threading.Thread(target=self.status_loop, daemon=True).start()

    def set_downvision(self, enabled):
        enabled = bool(enabled)
        if self.downvision_enabled == enabled:
            return
        response = self.tello.send_command_with_return(f"downvision {1 if enabled else 0}")
        if str(response).strip().lower() != "ok":
            raise CommandRejected(f"Camera wisselen mislukt: {response}")
        self.downvision_enabled = enabled
        self.pad_tracker.reset()
        self._live_pad = None
        print(f"Downvision {'enabled' if enabled else 'disabled'}")

    # ------------------------------------------------- jetson / manual input
    def submit(self, msg):
        if msg.get("type") == "flip":
            msg = dict(msg, _submitted=time.monotonic(),
                       _waiting_for=getattr(self, "_motion_phase", None)
                       or getattr(self, "_active_command", None))
            # Interrupt optional RC navigation before copying camera references;
            # the actual flip remains serialized by the mission worker.
            self.commands.flip_pending.set()
        # Click capture/queue insertion and idle landmark relocalization must
        # be atomic: never move the frame between capturing and queuing a target.
        try:
            with self._command_guard():
                self._submit(msg)
        finally:
            # A rejected/invalid request must not leave navigation interrupted.
            if msg.get("type") == "flip":
                with self.commands.mutex:
                    if not any(cmd["type"] == "flip" for cmd in self.commands.queue):
                        self.commands.flip_pending.clear()

    def _command_guard(self):
        if not hasattr(self, "_command_lock"):
            self._command_lock = threading.RLock()
        return self._command_lock

    def _submit(self, msg):
        """
        Entry point for every command, from the GUI or the Jetson.
        Runs in the caller's thread: never send flight commands from here.
        """
        t = msg["type"]
        if t in ("abort", "land"):
            if t == "abort":
                print("🛑 Abort")
            # Missions etc. queued before this Land/Abort must not start afterwards
            self._gen += 1
            self.abort.set()  # stop a running mission after its current step
            while True:
                try:
                    self.commands.get_nowait()
                except queue.Empty:
                    break
            self.commands.put({"type": "land", "_gen": self._gen})
        elif t == "ping":
            self.link.send({"type": "pong", "t": msg.get("t")})
        elif t in ("mission", "takeoff", "set_pose", "helipad_land", "reset", "downvision", "flip"):
            if t == "flip":
                snapshot = self.pose.snapshot()
                reference = None
                odometry = getattr(self, "odometry", None)
                if odometry is not None:
                    captured = odometry.capture_flip_reference()
                    if isinstance(captured, dict):
                        snapshot, reference = captured["snapshot"], captured["reference"]
                try:
                    target = tuple(float(v) for v in msg.get("pose", snapshot["pose"][:3])[:3])
                except (ValueError, TypeError):
                    self.link.send({"type": "error", "ref": "flip", "error": "invalid return target"})
                    return
                if len(target) != 3 or not all(math.isfinite(v) for v in target):
                    self.link.send({"type": "error", "ref": "flip", "error": "invalid return target"})
                    return
                record = {"captured_at": time.time(), "target_cm": target,
                          "source": "simulation" if getattr(self.tello, "is_simulator", False) is True else "drone",
                          "yaw_deg": snapshot["pose"][3],
                          "tracking_valid_at_click": snapshot["tracking_valid"],
                          "measured_at_click": snapshot["measured"],
                          "global_position_valid_at_click": snapshot["position_valid"],
                          "visual_reference_at_click": reference is not None,
                          "reference_quality": reference.get("quality") if reference else None,
                          "quality_at_click": snapshot["quality"],
                          "frame_id": getattr(self, "frame_id", 0), "status": "requested"}
                self.last_flip_record = record
                # Each queued flip owns its target; later clicks cannot replace it.
                msg = dict(msg, pose=target, _flip_record=record, _visual_reference=reference)
            self.commands.put(dict(msg, _gen=self._gen,
                                   _submitted=msg.get("_submitted", time.monotonic())))
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
            "position_valid": self.pose.position_valid,
            "mission": self.mission_id, "waypoint_index": self.waypoint_index,
            "helipad": self._helipad_msg(),
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
                    self.follower.rc(0, 0, 0, 0)
                    self._last_cmd_time = time.time()
                continue
            try:
                with self._command_guard():
                    self._active_command = cmd.get("type")
                self.handle_command(cmd)
            except FlightInterrupted:
                # The active navigation ends here. The queued flip runs next;
                # do not land or resume the interrupted mission afterwards.
                self.follower.rc(0, 0, 0, 0)
                self.link.send({"type": "flight_interrupted", "ref": cmd.get("type"),
                                "reason": "priority_flip"})
            except CommandRejected as e:   # nothing happened: don't land because of it
                print(f"⚠️  {cmd.get('type')} geweigerd: {e}")
                self.link.send({"type": "error", "ref": cmd.get("type"), "error": str(e)})
            except MissionAborted:
                print("🛑 Afgebroken")
                self.safe_land()
            except Exception as e:
                print(f"❌ {cmd.get('type')} failed: {e}")
                self.state = "error"
                self.link.send({"type": "error", "ref": cmd.get("type"), "error": str(e)})
                self.safe_land()
            finally:
                self.follower.level = False
                with self._command_guard():
                    self._active_command = None
                    self.commands.dispatching.clear()
            self._last_cmd_time = time.time()

    def _begin(self, cmd):
        """Start of a flight command: clear an old abort, unless Land/Abort came after it."""
        self.abort.clear()
        if cmd.get("_gen", self._gen) != self._gen:
            self.abort.set()
            raise MissionAborted()

    def handle_command(self, cmd):
        t = cmd["type"]
        if t == "mission":
            self.run_mission(cmd)
        elif t == "land":
            self.safe_land()
        elif t == "takeoff":
            self._begin(cmd)
            self.takeoff()
        elif t == "helipad_land":
            self._begin(cmd)
            self.helipad_test()
        elif t == "reset":
            self.reset_environment()
        elif t == "downvision":
            self.set_downvision(cmd.get("enabled", True))
        elif t == "flip":
            # Perform a flip in the mission thread, without a preparatory climb.
            self._begin(cmd)
            record = cmd.get("_flip_record")
            visual_reference = cmd.get("_visual_reference")
            direction = cmd.get("dir", "f")
            if direction not in ("l", "r", "f", "b"):
                raise CommandRejected("invalid flip direction")
            # remember previous position so we can return there after the flip
            try:
                prev = tuple(self.pose.get())
            except Exception:
                prev = None
            # Pose captured by the GUI at the moment the user clicked Salto.
            # Prefer this for the return target so the drone goes back to the
            # same place/height even if the queue waited a bit before running.
            press_pose = cmd.get("pose")
            if press_pose is not None:
                try:
                    px, py, pz = (float(v) for v in press_pose[:3])
                    press_pose = (px, py, pz)
                except Exception:
                    press_pose = None
            # The GUI sends the spot where the button was pressed; keep it for
            # the final recentering, but do not reposition to it before the flip.
            spot = cmd.get("spot")
            if spot:
                try:
                    press_target = (float(spot[0]), float(spot[1]))
                except Exception:
                    press_target = None
            else:
                press_target = None
            try:
                battery = int(self.tello.get_battery())
            except Exception:
                battery = None
            if battery is not None and battery < cfg.FLIP_MIN_BATTERY:
                raise CommandRejected(
                    f"battery too low for flip ({battery}% < {cfg.FLIP_MIN_BATTERY}%)"
                )
            # If we're not airborne, take off first so the flip is allowed by the SDK.
            if not self.airborne:
                try:
                    self.takeoff()
                except MissionAborted:
                    # takeoff was aborted (Land/Abort): stop handling flip
                    print("🛑 Flip aborted because takeoff was interrupted")
                    return
                # Takeoff may reset the coordinate frame. A ground pose is not
                # a valid airborne return target in that new frame.
                prev = tuple(self.pose.get())
                press_pose = press_target = None
                visual_reference = None
                if record is not None:
                    record["status"] = "ground_target_replaced_after_takeoff"
                    record["airborne_target_cm"] = prev[:3]
                    record["tracking_valid_at_click"] = False
                    record["visual_reference_at_click"] = False
            # Send at the current altitude as soon as the worker is available.
            # No extra RC command or artificial pause before the SDK command;
            # the firmware still decides whether current conditions allow it.
            self.follower._check_abort()
            # If the flip command fails we catch the error here and do NOT
            # re-raise — otherwise the mission loop treats it as a fatal
            # exception and calls safe_land(), which makes the drone descend.
            odometry = getattr(self, "odometry", None)
            self.local_pads = []
            self.local_known_pad = None
            if odometry is not None:
                odometry.begin_flip(visual_reference)
            else:
                self.pose.position_valid = False
            try:
                latency = time.monotonic() - cmd.get("_submitted", time.monotonic())
                print(f"Salto: SDK-verzoek na {max(0, latency) * 1000:.0f} ms")
                if record is not None:
                    record.update(dispatch_latency_ms=max(0, latency) * 1000,
                                  waiting_for=cmd.get("_waiting_for"))
                if latency > 0.5 and cmd.get("_waiting_for"):
                    print(f"ℹ️  Salto wachtte op: {cmd['_waiting_for']}; "
                          "een al verstuurde SDK-beweging moet eerst eindigen")
                # Exactly one request. djitellopy.flip() retries on errors/timeouts,
                # which can execute a delayed or repeated flip.
                response = self.tello.send_command_with_return(
                    f"flip {direction}", timeout=cfg.FLIP_ACK_TIMEOUT_S)
                if str(response).strip().lower() != "ok":
                    raise RuntimeError(f"flip: {response}")
            except Exception as e:
                # Log and notify the link, but avoid landing automatically.
                print(f"❌ flip failed (handled): {e}")
                try:
                    self.link.send({"type": "error", "ref": "flip", "error": str(e)})
                except Exception:
                    pass
            finally:
                if odometry is not None:
                    odometry.end_flip()
            # Use the same recovery path after success or an SDK error. The GUI
            # sends xyz, while Pose.get() returns xyz+yaw: normalize to xyz.
            self.follower.interruptible = True
            recovery_started = time.monotonic() if record is not None else None
            try:
                target = press_pose or prev
                if press_pose is None and press_target is not None:
                    target = (*press_target, (prev or self.pose.get())[2])
                if target:
                    px, py, pz = target[:3]
                    return_target = (px, py, float(pz))
                    if record is not None and not record["tracking_valid_at_click"] and visual_reference is None:
                        raise CommandRejected("Klikpositie opgeslagen, maar geen bruikbaar "
                                              "camerareferentie bij de klik; terugkeer niet bevestigd")
                    # Let video/height measurements catch up before deciding how
                    # to move. Land/Abort interrupts this wait and all recovery.
                    deadline = time.monotonic() + cfg.FLIP_TRACK_TIMEOUT_S + cfg.FRAME_LATENCY_S
                    def return_reference_valid():
                        return self.pose.position_valid or (visual_reference is not None and
                                                           self.pose.local_reference_valid)

                    while not (self.pose.measured and return_reference_valid()):
                        self.follower._check_abort()
                        if time.monotonic() >= deadline:
                            detail = odometry.recovery_status() if odometry is not None else {}
                            reason = detail.get("reason") if isinstance(detail, dict) else None
                            explanation = {
                                "not_enough_matching_features": "te weinig vaste herkenningspunten tussen beide beelden",
                                "not_enough_floor_features": "te weinig zichtbare vloertextuur",
                                "registered_images_disagree": "beelden komen niet betrouwbaar overeen",
                                "invalid_height": "geen bruikbare hoogtemeting",
                                "downward_camera_disabled": "ondercamera staat uit",
                            }.get(reason, "geen betrouwbare overeenkomst met het opgeslagen beeld")
                            raise CommandRejected("Klikreferentie opgeslagen, maar vloerbeeld na salto "
                                                  f"niet herkend ({explanation}); drone blijft hangen")
                        if self.abort.wait(0.05):
                            raise MissionAborted()
                    if record is not None:
                        record["tracking_wait_s"] = time.monotonic() - recovery_started
                    print(f"ℹ️  Terug naar vorige plek na salto ({px:.0f}, {py:.0f}, {pz:.0f})")
                    self.follower.interruptible = True
                    try:
                        self.follower._check_abort()
                        if self.pose.measured:
                            # RC recovery avoids the SDK settling wait and the
                            # stop between coarse and fine positioning.
                            arrived = self.follower.hold_position(
                                return_target, max_speed=cfg.FLIP_RETURN_RC_SPEED,
                                gain=cfg.FLIP_RETURN_GAIN,
                                tolerance_cm=cfg.FLIP_RETURN_TOL_CM,
                                timeout_s=cfg.FLIP_RETURN_TIMEOUT_S,
                                settle_s=cfg.FLIP_RETURN_SETTLE_S,
                                damping=cfg.FLIP_RETURN_DAMPING,
                                max_vertical_speed=cfg.FLIP_RETURN_Z_SPEED,
                                **({"allow_local_reference": True} if visual_reference is not None else {}))
                            current = self.pose.get()
                            xy_error = math.hypot(current[0] - px, current[1] - py)
                            z_error = abs(current[2] - pz)
                            confirmed = (arrived is True and self.pose.measured and return_reference_valid()
                                         and xy_error < cfg.FLIP_RETURN_TOL_CM
                                         and z_error < cfg.FLIP_RETURN_TOL_CM)
                            if record is not None:
                                record.update(status="confirmed" if confirmed else "not_confirmed",
                                              return_reference="click_image" if visual_reference is not None else "global",
                                              final_estimate_cm=current[:3], xy_error_cm=xy_error,
                                              z_error_cm=z_error)
                                control = getattr(self.follower, "last_hold_result", None)
                                if isinstance(control, dict):
                                    record["return_control"] = control
                            print(f"Salto terugkeer {'bevestigd' if confirmed else 'NIET bevestigd'}: "
                                  f"xy {xy_error:.1f} cm, hoogte {z_error:.1f} cm")
                    finally:
                        self.follower.interruptible = False
                    if not self.pose.measured:
                        print("⚠️  Terugkeer na salto: geen visuele odometrie; "
                              "werkelijke horizontale positie niet bevestigd")
            except (MissionAborted, FlightInterrupted):
                if record is not None:
                    record["status"] = "interrupted"
                raise
            except Exception as e:
                if record is not None:
                    record.update(status="not_confirmed", error=str(e))
                print(f"⚠️  Terugkeer na salto mislukt: {e}")
                self.link.send({"type": "error", "ref": "flip_return", "error": str(e)})
            finally:
                self.follower.interruptible = False
                if odometry is not None:
                    odometry.finish_flip_recovery()
                if record is not None:
                    record["recovery_s"] = time.monotonic() - recovery_started
                    self._save_flip_record(record)
        elif t == "set_pose":
            if self.airborne:
                raise CommandRejected("set_pose is only allowed on the ground")
            self.pose.set(cmd.get("x"), cmd.get("y"), None, cmd.get("yaw"))
            self._pose_set = True
            self.link.send({"type": "ack", "ref": t})

    def _save_flip_record(self, record):
        """Persist the click target and measured result after flight commands."""
        try:
            path = os.path.abspath(cfg.FLIP_RECORD_PATH)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            odometry = getattr(self, "odometry", None)
            if odometry is not None:
                diagnostic = odometry.save_flip_diagnostics(os.path.dirname(path))
                if isinstance(diagnostic, dict):
                    record["registration"] = diagnostic
            with open(path + ".tmp", "w", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
            os.replace(path + ".tmp", path)
        except (OSError, ValueError) as e:
            print(f"Salto-opname kon niet worden opgeslagen: {e}")

    def reset_frame(self):
        """
        New mission frame from where the drone stands now: origin here, x = where
        the nose (front camera) points, y = left. Everything measured in the old
        frame (H sightings, terrain map, GUI trail) no longer fits, so it goes too.
        """
        self.pose.reset()
        if hasattr(self, "odometry"):
            self.odometry.reset_tracking()
        self.pads = []
        self.known_pad = None
        self.local_pads = []
        self.local_known_pad = None
        self.ever_seen_h = False
        self._h_relocalization = []
        self._live_pad = None
        self.pad_tracker.reset()
        if hasattr(self, "telemetry"):
            self.telemetry.terrain.reset()
        self.frame_id += 1

    def reset_environment(self):
        """Reset button: new frame and forget the last mission. Only on the ground."""
        if self.airborne:
            raise CommandRejected("reset is only allowed on the ground")
        self.reset_frame()
        self._pose_set = False
        self.abort.clear()
        self.mission_id = None
        self.current_waypoints = []
        self.waypoint_index = -1
        self.waypoints_reached = 0
        self.state = "idle"
        print("🔄 Omgeving gereset: drone staat op (0, 0), x = richting van de neus")
        self.link.send({"type": "ack", "ref": "reset"})

    def takeoff(self):
        if self.airborne:
            return
        battery = self.tello.get_battery()
        if battery < cfg.MIN_BATTERY:
            raise RuntimeError(f"battery too low ({battery}% < {cfg.MIN_BATTERY}%)")
        if cfg.RESET_FRAME_ON_TAKEOFF and not self._pose_set:
            self.reset_frame()   # coordinates of this flight start at the drone itself
        self.state = "taking_off"
        print("🚁 Opstijgen...")
        x0, y0, _, _ = self.pose.get()
        self.airborne = True  # odometry measures the drift during takeoff too
        self._motion_phase = "takeoff_sdk"
        try:
            self.tello.takeoff()
        except Exception:
            self.airborne = False
            raise
        finally:
            self._motion_phase = None
        height = self.read_height(default=80)
        self.pose.set(z=height)
        self.state = "flying"
        # Climbing changes image scale and can temporarily interrupt tracking.
        # Allow fresh level frames to arrive before deciding whether to recenter.
        self.follower.interruptible = True
        self._motion_phase = "takeoff_recenter"
        try:
            deadline = time.monotonic() + 0.8
            while not self.pose.measured and time.monotonic() < deadline:
                self.follower._check_abort()
                if self.abort.wait(0.05):
                    raise MissionAborted()
            if cfg.TAKEOFF_RECENTER and self.pose.measured and self.pose.position_valid:
                x, y, _, _ = self.pose.get()
                if math.hypot(x - x0, y - y0) > 3:
                    print("↩️  Terug boven de startplek")
                    self.follower.hold_position((x0, y0, height))
        finally:
            self.follower.interruptible = False
            self._motion_phase = None
        x, y, _, _ = self.pose.get()
        print(f"✅ In de lucht op {height} cm ({x - x0:+.0f}, {y - y0:+.0f} cm verschoven)")

    def helipad_test(self):
        """Test without a path: take off (if needed), look for the H here and land on it."""
        self.takeoff()
        x, y, _, _ = self.pose.get()
        local = not self.pose.position_valid and self.pose.local_reference_valid
        remembered = self.seen_helipad(local=local)
        live = self._live_pad
        visible = live is not None and time.time() - live[0] < cfg.HELIPAD_CONTROL_MAX_AGE_S
        if not visible and remembered is not None and (self.pose.position_valid or local) and self.pose.measured:
            print(f"H onthouden op ({remembered[0]:.0f}, {remembered[1]:.0f})")
            try:
                if local:
                    distance = math.hypot(remembered[0] - x, remembered[1] - y)
                    timeout = min(cfg.HELIPAD_APPROACH_MAX_S, max(cfg.FINE_TIMEOUT_S, distance / 20 * 2 + 3))
                    arrived = self.follower.hold_position((*remembered, self.pose.get()[2]),
                                                         allow_local_reference=True,
                                                         damping=cfg.HELIPAD_DAMPING, timeout_s=timeout)
                    if not arrived:
                        live = self._live_pad
                        if live is None or time.time() - live[0] >= cfg.HELIPAD_CONTROL_MAX_AGE_S:
                            raise CommandRejected("Lokale H-positie niet bereikt en H niet in beeld: drone blijft hangen")
                else:
                    self.follower.move_to((*remembered, self.pose.get()[2]), cfg.DEFAULT_SPEED,
                                          require_tracking=True)
            except OdometryLost as e:
                raise CommandRejected("Positie kwijt: terugkeer naar onthouden H gestopt") from e
            x, y = remembered
        print("🔎 H-test: zoeken naar een H onder de drone")
        found = False
        try:
            found = self.land_on_helipad((x, y))
        except MissionAborted:
            self.safe_land(helipad=False)
            return
        if not found:
            raise CommandRejected("H niet bevestigd: geen automatische landing; "
                                  "drone blijft hangen. Landen (L) voor hier landen.")
        self.safe_land(helipad=False)

    def safe_land(self, spot=None, helipad=True):
        """
        Land. With spot=(x, y) and working odometry: precise landing on that spot,
        or on an H landing pad near it (helipad and HELIPAD_LAND).
        """
        if not self.airborne:
            self.state = "idle"
            return
        self.state = "landing"
        print("🛬 Landen...")
        if spot is not None and helipad and cfg.HELIPAD_LAND:
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
        for attempt in range(2):
            try:
                self.follower.command(self.tello.land)
                break
            except Exception as e:
                print(f"⚠️  Land command failed: {e}")
        else:
            if self.read_height(default=0) > 30:
                # still in the air: stay 'airborne' (keepalive, Land button works again)
                print("❌ Landen mislukt: de drone hangt nog. Druk opnieuw op Landen.")
                self.state = "error"
                return
        self.airborne = False
        self._pose_set = False
        self.pose.set(z=0)
        self.state = "idle"

    def run_mission(self, msg):
        # check everything before taking off; a bad mission must not land a hovering drone
        try:
            waypoints = validate_waypoints(msg["waypoints"])
            speed = int(round(float(msg.get("speed", cfg.DEFAULT_SPEED))))
        except (KeyError, TypeError, ValueError) as e:
            raise CommandRejected(f"invalid mission: {e}")
        if not 10 <= speed <= 100:
            raise CommandRejected(f"speed {speed} outside 10-100 cm/s")
        land_at_end = msg.get("land_at_end", cfg.LAND_AT_END)
        # level: keep the height after takeoff, don't follow the floor (ToF)
        # helipad_search: land on an H seen anywhere on the way, else on the first point
        self.follower.level = bool(msg.get("level", False))
        helipad_search = bool(msg.get("helipad_search", False))
        self.mission_id = msg.get("id")
        self.current_waypoints = waypoints
        self.waypoints_reached = 0
        self._begin(msg)
        print(f"🗺️  Mission {self.mission_id}: {len(waypoints)} waypoints @ {speed} cm/s")
        self.link.send({"type": "ack", "ref": "mission", "id": self.mission_id,
                        "waypoints": len(waypoints)})
        try:
            self.takeoff()
        except MissionAborted:   # Land/Abort during takeoff
            print("🛑 Mission aborted")
            self.link.send({"type": "mission_aborted", "id": self.mission_id,
                            "helipad": self._helipad_msg()})
            self.safe_land()
            return

        def on_waypoint(i, wp):
            self.waypoint_index = i
            self.waypoints_reached = i + 1
            x, y, z, _ = self.pose.get()
            print(f"📍 Waypoint {i + 1}/{len(waypoints)} bereikt ({x:.0f}, {y:.0f}, {z:.0f})")
            self.link.send({"type": "waypoint_reached", "id": self.mission_id, "index": i,
                            "pos": {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1)}})

        try:
            self.follower.interruptible = True
            self.follower.fly(waypoints, speed, on_waypoint,
                              mode=msg.get("nav_mode"), fine=msg.get("fine"))
        except MissionAborted:
            print("🛑 Mission aborted")
            self.link.send({"type": "mission_aborted", "id": self.mission_id,
                            "helipad": self._helipad_msg()})
            self.safe_land()
            return
        finally:
            self.follower.interruptible = False
            self.waypoint_index = -1

        print("🏁 Mission complete")
        self.link.send({"type": "mission_done", "id": self.mission_id,
                        "helipad": self._helipad_msg()})
        try:
            if land_at_end and cfg.HELIPAD_LAND and self.seen_helipad() is not None:
                # an H seen anywhere on the way is the landing spot
                self.land_on_seen_helipad(waypoints[0], speed)
            elif land_at_end and helipad_search:
                self.land_on_seen_helipad(waypoints[0], speed)   # no H: first point
            elif land_at_end:
                self.safe_land(spot=waypoints[-1][:2])   # (still looks for an H there)
        except MissionAborted:
            self.safe_land()

    def seen_helipad(self, local=False):
        """
        Position of the H seen during this mission, or None. The sightings are
        grouped by place; the place seen most often (at least HELIPAD_MIN_SIGHTINGS
        times, so one wrong detection doesn't count) wins, its median is returned.
        """
        pads = list(getattr(self, "local_pads", []) if local else self.pads)
        known = getattr(self, "local_known_pad", None) if local else self.known_pad
        if known is not None:
            near = [p for p in pads if math.hypot(p[1] - known[0], p[2] - known[1]) < 30]
            if len(near) >= cfg.HELIPAD_MIN_SIGHTINGS:
                xs, ys = sorted(p[1] for p in near), sorted(p[2] for p in near)
                known = (xs[len(xs) // 2], ys[len(ys) // 2])
                setattr(self, "local_known_pad" if local else "known_pad", known)
            return known
        best = []
        for _, x, y, _ in pads:
            near = [p for p in pads if math.hypot(p[1] - x, p[2] - y) < 30]
            if len(near) > len(best):
                best = near
        if len(best) < cfg.HELIPAD_MIN_SIGHTINGS:
            return None
        xs = sorted(p[1] for p in best)
        ys = sorted(p[2] for p in best)
        known = (xs[len(xs) // 2], ys[len(ys) // 2])
        setattr(self, "local_known_pad" if local else "known_pad", known)
        return known

    def _helipad_msg(self):
        pad = self.seen_helipad()
        return None if pad is None else {"x": round(pad[0], 1), "y": round(pad[1], 1)}

    def _try_h_relocalization(self, t_frame, x, y):
        """Recover the original frame only from a previously known, stable H.

        A newly discovered local H cannot establish where the flight began.
        Don't shift coordinates while a flight command is using a saved target.
        """
        if (getattr(self, "_active_command", None) is not None or self.known_pad is None
                or self.pose.position_valid or not self.pose.measured
                or not self.pose.local_reference_valid or not hasattr(self, "odometry")):
            self._h_relocalization = []
            return
        delta = (self.known_pad[0] - x, self.known_pad[1] - y)
        if math.hypot(*delta) > cfg.HELIPAD_RELOCALIZE_MAX_CM:
            self._h_relocalization = []
            return
        samples = getattr(self, "_h_relocalization", [])
        samples = [s for s in samples if t_frame - s[0] < 0.7]
        if samples and t_frame <= samples[-1][0]:
            return
        samples.append((t_frame, *delta))
        self._h_relocalization = samples[-cfg.HELIPAD_MIN_SIGHTINGS:]
        if len(self._h_relocalization) < cfg.HELIPAD_MIN_SIGHTINGS:
            return
        samples = self._h_relocalization
        dx = sorted(s[1] for s in samples)[len(samples) // 2]
        dy = sorted(s[2] for s in samples)[len(samples) // 2]
        if max(math.hypot(s[1] - dx, s[2] - dy) for s in samples) > cfg.HELIPAD_RELOCALIZE_SPREAD_CM:
            return
        with self._command_guard():
            if (self._active_command is not None or self.commands.dispatching.is_set()
                    or not self.commands.empty()):
                return
            if self.odometry.relocalize_from_pad(dx, dy) is True:
                print("📍 Globale positie hersteld via de eerder onthouden H")
                self.local_pads = []
                self.local_known_pad = None
                self._h_relocalization = []

    def land_on_seen_helipad(self, first_wp, speed):
        """
        End of a mission: fly back to where the H was seen and land on it. No H
        seen during the whole flight (helipad_search): fly back to the first
        waypoint and land there.
        """
        z = self.pose.get()[2]
        pad = self.seen_helipad()
        if pad is not None:
            print(f"🛬 H gezien tijdens de vlucht: terug naar ({pad[0]:.0f}, {pad[1]:.0f})")
            self.follower.move_to((pad[0], pad[1], z), speed)
            self.safe_land(spot=pad)
        else:
            print("ℹ️  Geen H gezien tijdens de vlucht: landen op het eerste punt")
            self.follower.move_to((first_wp[0], first_wp[1], z), speed)
            self.safe_land(spot=first_wp[:2], helipad=False)

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
        t_climb = time.time() + 8      # climb at most this long
        try:
            while time.time() < t_end:
                self.follower._check_abort()
                found = self._pad_near(end_xy, 0)
                live = self._live_pad
                if live is not None and time.time() - live[0] < 0.4:
                    x, y, _, _ = self.pose.get()
                    return (live[0], x + live[1], y + live[2], live[3])
                if found and time.time() - found[0] < 0.4:
                    return found
                x, y, z, _ = self.pose.get()
                vz = 0
                if (climb_cms and self.pose.measured and self.pose.position_valid
                        and z < cfg.HELIPAD_SEARCH_HEIGHT_CM and time.time() < t_climb):
                    vz = climb_cms
                    t_end = time.time() + cfg.HELIPAD_SEARCH_S   # look once up there
                # stay above the end point (otherwise the drone drifts away while looking);
                # without odometry the position doesn't update: steering would fly away
                vx, vy = (cfg.RC_GAIN * (e - c) for e, c in zip(end_xy, (x, y)))
                if not self.pose.measured or not self.pose.position_valid:
                    vx = vy = 0.0
                n = math.hypot(vx, vy)
                if n > 15:
                    vx, vy = vx * 15 / n, vy * 15 / n
                self.follower.send_velocity(vx, vy, vz)
                time.sleep(0.1)
            return None
        finally:
            self.follower.rc(0, 0, 0, 0)

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
        if sighting is None or time.time() - sighting[0] > 0.4:
            sighting = self._look_for_pad(end_xy, 0.0)
        if (sighting is None and self.pose.measured and self.pose.position_valid
                and self.pose.get()[2] < cfg.HELIPAD_SEARCH_HEIGHT_CM - 15):
            # higher up the camera sees a bigger part of the floor
            print(f"🔎 Geen H in beeld: stijgen naar {cfg.HELIPAD_SEARCH_HEIGHT_CM} cm om te zoeken")
            sighting = self._look_for_pad(end_xy, cfg.HELIPAD_CLIMB_CMS)
        if sighting is None:
            if not self.pose.measured or not self.pose.position_valid:
                print("⚠️  H buiten beeld en positie onbekend: niet gestegen of blind teruggevlogen. "
                      "De onthouden H-coördinaten zijn bewaard, maar niet bereikbaar zonder lokalisatie.")
            else:
                print("⚠️  Geen H in beeld: landing op H niet bevestigd")
            return False
        t_seen, tx, ty, size = sighting
        print(f"🛬 H gevonden op ({tx:.0f}, {ty:.0f}): erboven centreren en dalen")
        period = 1.0 / cfg.RC_HZ
        deadline = time.time() + cfg.HELIPAD_TIMEOUT_S
        centred_since = None
        motion = MotionEstimate()
        try:
            while time.time() < deadline:
                f._check_abort()
                live = self._live_pad
                if live is not None:
                    t_seen, ex, ey, size, _ = live
                else:
                    ex = ey = 0.0
                x, y, z, _ = self.pose.get()
                err = math.hypot(ex, ey)
                lost = time.time() - t_seen
                if live is None or lost > cfg.HELIPAD_CONTROL_MAX_AGE_S:
                    # Stale pixel offsets must never keep pushing the drone.
                    f.rc(0, 0, 0, 0)
                    centred_since = None
                    if lost > 3.0:
                        print("⚠️  H uit beeld: landing onderbroken")
                        return False
                    time.sleep(period)
                    continue
                # low enough, or the H almost fills the image (lower it would
                # get cut off by the image edge and can't be seen any more)
                big = size >= cfg.HELIPAD_FINAL_SIZE and lost < 1.0
                centred = err < cfg.HELIPAD_CENTER_TOL_CM * (2 if z > 60 else 1)
                if centred:
                    centred_since = time.time() if centred_since is None else centred_since
                else:
                    centred_since = None
                stable = (centred_since is not None and
                          time.time() - centred_since >= cfg.HELIPAD_CENTER_STABLE_S)
                if stable and err < cfg.HELIPAD_CENTER_TOL_CM and (z <= cfg.HELIPAD_FINAL_CM or big):
                    print(f"🎯 Boven het midden van de H ({err:.0f} cm ernaast)")
                    return True
                # only come down while centred and the H is in view; otherwise
                # hold the height and steer back above it
                vz = -cfg.HELIPAD_DESCENT_CMS if stable else 0.0
                vx_measured, vy_measured, _ = motion.observe((-ex, -ey, 0), t_seen)
                vx, vy = (cfg.RC_GAIN * ex - cfg.HELIPAD_DAMPING * vx_measured,
                          cfg.RC_GAIN * ey - cfg.HELIPAD_DAMPING * vy_measured)
                n = math.hypot(vx, vy)
                if n > 15:
                    vx, vy = vx * 15 / n, vy * 15 / n
                if not self.pose.measured and lost > 0.5:
                    # no odometry and no fresh view of the H: the position doesn't
                    # update, steering on it would fly the drone away. Hold still.
                    vx = vy = 0.0
                f.send_velocity(vx, vy, vz, min_units=0 if centred else cfg.RC_MIN_UNITS)
                time.sleep(period)
            print(f"⚠️  H: na {cfg.HELIPAD_TIMEOUT_S}s niet gecentreerd: landing onderbroken")
            return False
        finally:
            self.follower.rc(0, 0, 0, 0)

    def read_height(self, default):
        """Height above the floor from the ToF sensor (cm), fallback to default."""
        try:
            h = self.tello.get_distance_tof()
            if 10 < h < 1000:
                return h
        except Exception:
            pass
        return default

    # ---------------------------------------------------------- camera thread
    def detect_loop(self):
        period = 1.0 / cfg.DETECT_HZ
        previous = None
        while self.running:
            t_start = time.time()
            frame = self.frame_reader.frame
            if frame is not None and frame is not previous:
                previous = frame
                try:
                    self.process_frame(frame, t_start)
                except Exception as e:   # never let the camera view freeze
                    if str(e) != getattr(self, "_detect_error", None):
                        self._detect_error = str(e)
                        print(f"⚠️  Fout bij het verwerken van het camerabeeld: {e!r}")
                        traceback.print_exc(file=sys.stdout)
            time.sleep(max(0.0, period - (time.time() - t_start)))

    def emergency_stop(self):
        """Motors off immediately. The drone falls!"""
        print("🚨 EMERGENCY STOP")
        self._gen += 1   # nothing queued may start after this
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
        """Last processed downward image with the H drawn (BGR), or None."""
        with self._vis_lock:
            return self._vis

    def process_frame(self, frame, t_frame):
        pose = self.pose.get(t_frame - cfg.FRAME_LATENCY_S)
        gray = self.cam.prepare(frame)
        if self.recording:
            self.record_frame(gray)
        # Also on the ground, so you can test it by holding the drone above the H
        # (it is drawn green in the GUI camera view)
        # (only on the downward camera: an H on a wall is no landing pad)
        height = self.read_height(default=None)   # ToF, also when held in the hand
        if self.airborne and pose[2] >= 15:
            height = pose[2]  # same capture-time scale for detection and geometry
        level = not hasattr(self, "odometry") or (self.odometry.level() and not self.odometry._flip_active)
        if cfg.HELIPAD_LAND and self.downvision_enabled and level:
            candidates = self.pad_detector.find_candidates(gray, height)
            pad = self.pad_tracker.update(candidates, t_frame, gray.shape)
        else:
            self.pad_tracker.reset()
            self._live_pad = None
            pad = None
        fresh = pad is not None and self.pad_tracker.last_seen == t_frame
        if fresh and self.airborne:
            self.ever_seen_h = True
            # Use capture-time height, not current ToF during a climb/descent.
            height = pose[2] if pose[2] >= 15 else height
            if height >= 15:
                h, w = gray.shape
                x, y, _ = pixel_to_world(pad.cx, pad.cy, w, h, height, pose)
                dx, dy, _ = pixel_to_world(pad.cx, pad.cy, w, h, height,
                                          (0, 0, pose[2], pose[3]))
                self._live_pad = (t_frame, dx, dy, pad.size_px / w, height)
                if not (self.pose.measured and (self.pose.position_valid or self.pose.local_reference_valid)):
                    # Relative centering is still possible, global pad coordinates
                    # must not be memorized from an unknown flight position.
                    with self._vis_lock:
                        self._frame = frame.copy()
                        self._vis = draw_helipad(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), pad)
                    return
                local = not self.pose.position_valid
                history = self.local_pads if local else self.pads
                if not history or t_frame - history[-1][0] > 5:
                    print(f"🛬 H gezien op ({x:.0f}, {y:.0f})")
                history.append((t_frame, x, y, pad.size_px / w))
                del history[:-200]
                self.seen_helipad(local=local)
                if local:
                    self._try_h_relocalization(t_frame, x, y)
        vis = draw_helipad(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), pad, stale=not fresh)
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
