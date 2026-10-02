"""
Autonomous Tello EDU: follow a path (from the Jetson or entered manually) and
report puddles seen by the downward camera.

    python tello_autonomous.py                    # real drone, Jetson + manual console
    python tello_autonomous.py --waypoints "100,0,80; 100,100,80; 0,0,80"
    python tello_autonomous.py --mission pad.json # real drone, local mission file
    python tello_autonomous.py --record dataset   # save downward frames for Roboflow
    python tello_autonomous.py --sim              # simulator, no drone needed

Without a Jetson, type coordinates in the terminal (type 'help'), see manual_input.py.

Keys in the video window:
    L / P  land (aborts the running mission)
    R      start/stop recording downward frames (dataset)
    X      EMERGENCY: motors off immediately (drone falls!)
    ESC    land and quit

Threads:
    main      video window + keys + status messages
    mission   the ONLY thread that sends flight commands to the Tello
    detector  puddle detection on the downward camera
    link      receives Jetson messages (jetson_link.py)
    console   manual commands typed in the terminal (manual_input.py)
"""

import argparse
import json
import os
import queue
import threading
import time

import cv2

import config as cfg
from jetson_link import JetsonLink
from manual_input import ManualConsole, parse_point
from navigator import MissionAborted, PathFollower, Pose, validate_waypoints
from puddle_detector import PuddleTracker, create_detector, draw_detections, pixel_to_world


class DroneApp:
    def __init__(self, tello, mission_file=None, record_dir=None):
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
        if mission_file:
            with open(mission_file) as f:
                self.commands.put(json.load(f))

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
        Entry point for every command, from the Jetson or the manual console.
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
            "pos": {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1), "yaw": yaw},
            "mission": self.mission_id, "waypoint_index": self.waypoint_index,
            "puddles": len(self.tracker.confirmed()),
        })

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
        self.tello.takeoff()
        self.airborne = True
        height = self.read_height(default=80)
        self.pose.set(z=height)
        self.state = "flying"
        print(f"✅ In de lucht op {height} cm")

    def safe_land(self):
        if not self.airborne:
            self.state = "idle"
            return
        self.state = "landing"
        print("🛬 Landen...")
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
            self.follower.fly(waypoints, speed, on_waypoint)
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
            self.safe_land()

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

    def process_frame(self, frame, t_frame):
        pose = self.pose.get(t_frame - cfg.FRAME_LATENCY_S)
        dets, gray, _ = self.detector.detect(frame)
        if self.recording:
            self.record_frame(gray)
        height = self.read_height(default=pose[2])
        active = self.airborne and self.state == "flying" and height >= cfg.MIN_DETECT_HEIGHT
        if active:
            h, w = gray.shape
            for d in dets:
                x, y, s = pixel_to_world(d.cx, d.cy, w, h, height, pose)
                report = self.tracker.add(x, y, d.area_px * s * s)
                if report:
                    print(f"💧 Plas #{report['id']} op x={report['x']} y={report['y']} "
                          f"({report['area_cm2']:.0f} cm²)")
                    self.link.send({"type": "puddle", "mission": self.mission_id, **report})
        vis = draw_detections(gray, dets if active else [])
        with self._vis_lock:
            self._frame = frame.copy() if hasattr(frame, "copy") else frame
            self._vis = vis

    # ------------------------------------------------------------ main thread
    def run_ui(self):
        last_status = 0
        try:
            while True:
                with self._vis_lock:
                    vis = None if self._vis is None else self._vis.copy()
                if vis is not None:
                    vis = cv2.resize(vis, (640, 480))
                    x, y, z, _ = self.pose.get()
                    lines = [f"State: {self.state}  Battery: {self.tello.get_battery()}%",
                             f"Pos: x={x:.0f} y={y:.0f} z={z:.0f} cm",
                             f"Puddles: {len(self.tracker.confirmed())}"
                             + (f"  REC {self._record_count}" if self.recording else "")]
                    for i, text in enumerate(lines):
                        cv2.putText(vis, text, (10, 25 + 25 * i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Tello autonomous (downvision)", vis)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:                      # ESC
                    print("Exiting...")
                    break
                if key in (ord("l"), ord("p")):
                    self.submit({"type": "land"})
                if key == ord("r"):
                    self.toggle_recording()
                if key == ord("x"):
                    self.emergency_stop()
                if time.time() - last_status > cfg.STATUS_INTERVAL:
                    self.send_status()
                    last_status = time.time()
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Cleaning up...")
        self.abort.set()
        self.running = False
        # Wait until the mission thread finished its current move, so only one
        # thread talks to the Tello at a time.
        self.mission_thread.join(timeout=cfg.RESPONSE_TIMEOUT)
        self.safe_land()
        try:
            self.tello.streamoff()
        except Exception:
            pass
        self.link.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Drone landed and disconnected")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="use the simulator instead of a drone")
    parser.add_argument("--mission", help="JSON file with a mission (same format as from the Jetson)")
    parser.add_argument("--waypoints", help='manual path in cm, e.g. "100,0,80; 100,100,80; 0,0,80"')
    parser.add_argument("--speed", type=int, default=cfg.DEFAULT_SPEED, help="cm/s for --waypoints")
    parser.add_argument("--hover", action="store_true", help="don't land after --waypoints")
    parser.add_argument("--record", metavar="DIR", help="save downward frames to DIR (dataset)")
    parser.add_argument("--no-console", action="store_true", help="disable the manual terminal console")
    args = parser.parse_args()

    if args.sim:
        from sim import FakeTello
        tello = FakeTello()
    else:
        from djitellopy import Tello
        Tello.RESPONSE_TIMEOUT = cfg.RESPONSE_TIMEOUT
        tello = Tello()

    app = DroneApp(tello, args.mission, args.record)
    if args.waypoints:
        points = [p for p in args.waypoints.split(";") if p.strip()]
        waypoints = validate_waypoints([parse_point([p]) for p in points])
        app.submit({"type": "mission", "id": "cli", "speed": args.speed,
                    "land_at_end": not args.hover, "waypoints": waypoints})
    app.start()
    if not args.no_console:
        ManualConsole(app)
    app.run_ui()


if __name__ == "__main__":
    main()
