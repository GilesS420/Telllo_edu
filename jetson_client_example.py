"""
Example of the Jetson side: send a path to the drone laptop and receive puddles.

Replace world_to_drone() with your own conversion (e.g. metres in the room or
camera-based coordinates -> cm in the drone's mission frame).

    python jetson_client_example.py --drone 192.168.1.50
"""

import argparse
import json
import math
import socket
import time

DRONE_PORT = 9000   # = config.LISTEN_PORT on the drone laptop
LOCAL_PORT = 9001   # = config.JETSON_PORT

# Pose of the drone's start point in the real world (metres, degrees).
# The drone's mission frame: x = forward from start, y = left, z = up, in cm.
START_X_M, START_Y_M, START_HEADING_DEG = 0.0, 0.0, 0.0


def world_to_drone(xw, yw, zw):
    """Real-world metres -> drone mission frame in cm."""
    dx, dy = xw - START_X_M, yw - START_Y_M
    a = math.radians(START_HEADING_DEG)
    x = dx * math.cos(a) + dy * math.sin(a)
    y = -dx * math.sin(a) + dy * math.cos(a)
    return {"x": round(x * 100), "y": round(y * 100), "z": round(zw * 100)}


def drone_to_world(x_cm, y_cm):
    """Drone mission frame (cm) -> real-world metres."""
    x, y = x_cm / 100, y_cm / 100
    a = math.radians(START_HEADING_DEG)
    return (START_X_M + x * math.cos(a) - y * math.sin(a),
            START_Y_M + x * math.sin(a) + y * math.cos(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone", default="127.0.0.1", help="IP of the laptop that runs tello_autonomous.py")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", LOCAL_PORT))
    sock.settimeout(1.0)

    # Real-world path in metres (x, y, height)
    path_world = [(1.0, 0.0, 0.8), (2.0, 0.0, 0.8), (2.0, 1.2, 0.8), (0.5, 1.5, 0.8), (0.0, 0.0, 0.8)]
    mission = {"type": "mission", "id": f"m{int(time.time())}",
               "speed": 30, "land_at_end": True,
               "waypoints": [world_to_drone(*p) for p in path_world]}
    sock.sendto(json.dumps(mission).encode(), (args.drone, DRONE_PORT))
    print("Mission sent:", mission["waypoints"])

    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            sock.sendto(json.dumps({"type": "abort"}).encode(), (args.drone, DRONE_PORT))
            break
        msg = json.loads(data)
        if msg["type"] == "puddle":
            xw, yw = drone_to_world(msg["x"], msg["y"])
            print(f"💧 Puddle {msg['id']}: drone ({msg['x']}, {msg['y']}) cm "
                  f"-> world ({xw:.2f}, {yw:.2f}) m, ~{msg['area_cm2']:.0f} cm²")
        elif msg["type"] in ("mission_done", "mission_aborted"):
            print(f"{msg['type']}: {len(msg['puddles'])} puddle(s)")
            for p in msg["puddles"]:
                print("   ", p)
            break
        elif msg["type"] != "status":
            print(msg)


if __name__ == "__main__":
    main()
