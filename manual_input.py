"""
Terminal console to fly with manually entered coordinates (no Jetson needed).

Type commands in the terminal while tello_autonomous.py runs. The console
builds the same messages the Jetson would send, so the flight code is identical.
"""

import threading

import config as cfg
from navigator import validate_waypoints

HELP = """
Handmatige besturing (coördinaten in cm: x = vooruit, y = links, z = hoogte)
  100 0 80        waypoint toevoegen (ook: 100,0,80)
  list            waypoints tonen        undo   laatste weg     clear   alles weg
  start           waypoints afvliegen en daarna landen
  start hover     waypoints afvliegen en blijven hangen
  goto 100 0 80   direct naar één punt vliegen en blijven hangen
  speed 30        snelheid in cm/s (10-100)
  takeoff / land  opstijgen / landen (land breekt een missie af)
  pos             huidige positie        puddles  gevonden plassen
  record          opnemen van camerabeelden aan/uit (dataset voor Roboflow)
  help            deze uitleg
"""


def parse_point(parts):
    """'100 0 80' or '100,0,80' -> (x, y, z)."""
    values = " ".join(parts).replace(",", " ").replace(";", " ").split()
    if len(values) != 3:
        raise ValueError("geef 3 getallen: x y z")
    return tuple(float(v) for v in values)


class ManualConsole:
    def __init__(self, app):
        self.app = app
        self.waypoints = []
        self.speed = cfg.DEFAULT_SPEED
        self._count = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        print(HELP)
        while self.app.running:
            try:
                line = input("tello> ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not line:
                continue
            try:
                self.handle(line)
            except (ValueError, KeyError) as e:
                print(f"⚠️  {e}")

    def _mission(self, waypoints, land_at_end):
        self._count += 1
        return {"type": "mission", "id": f"manual-{self._count}", "speed": self.speed,
                "land_at_end": land_at_end, "waypoints": [list(p) for p in waypoints]}

    def handle(self, line):
        parts = line.split()
        cmd = parts[0].lower()
        app = self.app
        if cmd in ("help", "?", "h"):
            print(HELP)
        elif cmd[0].isdigit() or cmd[0] in "-+.":
            point = parse_point(parts)
            validate_waypoints([point])  # geofence check right away
            self.waypoints.append(point)
            print(f"  waypoint {len(self.waypoints)}: {point}")
        elif cmd == "list":
            if not self.waypoints:
                print("  (geen waypoints)")
            for i, p in enumerate(self.waypoints, 1):
                print(f"  {i}: x={p[0]:.0f} y={p[1]:.0f} z={p[2]:.0f}")
        elif cmd == "undo":
            if self.waypoints:
                print(f"  verwijderd: {self.waypoints.pop()}")
        elif cmd == "clear":
            self.waypoints = []
            print("  waypoints gewist")
        elif cmd == "start":
            if not self.waypoints:
                raise ValueError("eerst waypoints toevoegen")
            hover = len(parts) > 1 and parts[1].lower() == "hover"
            app.submit(self._mission(self.waypoints, land_at_end=not hover))
            self.waypoints = []
        elif cmd == "goto":
            point = parse_point(parts[1:])
            app.submit(self._mission([point], land_at_end=False))
        elif cmd == "speed":
            self.speed = int(min(max(int(parts[1]), 10), 100))
            print(f"  snelheid: {self.speed} cm/s")
        elif cmd in ("takeoff", "land"):
            app.submit({"type": cmd})
        elif cmd == "pos":
            x, y, z, yaw = app.pose.get()
            print(f"  x={x:.0f} y={y:.0f} z={z:.0f} yaw={yaw:.0f}  ({app.state})")
        elif cmd == "puddles":
            puddles = app.tracker.confirmed()
            if not puddles:
                print("  (nog geen plassen)")
            for p in puddles:
                print(f"  #{p['id']}: x={p['x']} y={p['y']} ~{p['area_cm2']:.0f} cm²")
        elif cmd == "record":
            app.toggle_recording()
        else:
            raise ValueError(f"onbekend commando '{cmd}' (typ help)")
