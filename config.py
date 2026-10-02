"""
Central configuration for the autonomous Tello EDU mission.

All distances are in centimetres, angles in degrees, unless stated otherwise.

Coordinate frame used everywhere (the "mission frame"):
    x = forward (drone nose direction at takeoff, unless set_pose sets a yaw)
    y = left
    z = height above the floor
    origin = where the drone stands when the script starts (or set via set_pose)
The Jetson converts real-world coordinates to this frame before sending them,
and converts the puddle coordinates it gets back to real-world coordinates.
"""

# ---------------------------------------------------------------------------
# Network link with the Jetson (JSON over UDP, one message per datagram)
# ---------------------------------------------------------------------------
LISTEN_HOST = "0.0.0.0"     # interface on which we accept commands
LISTEN_PORT = 9000          # Jetson sends commands to this port
JETSON_HOST = None          # None = reply to whoever sent the last command
JETSON_PORT = 9001          # Jetson listens on this port for puddles/status
STATUS_INTERVAL = 1.0       # seconds between status messages

# ---------------------------------------------------------------------------
# Flight
# ---------------------------------------------------------------------------
DEFAULT_SPEED = 30          # cm/s for 'go' commands (10-100)
MAX_STEP_CM = 100           # long segments are split in steps of max this length
                            # (smaller = more accurate puddle positions + faster abort)
MIN_MOVE_CM = 20            # Tello cannot 'go' when |x|,|y|,|z| are all < 20
MIN_BATTERY = 25            # refuse a mission below this battery %
RESPONSE_TIMEOUT = 20       # s, djitellopy wait time for 'ok' (long go commands)
KEEPALIVE_INTERVAL = 5      # s, Tello lands by itself after 15 s without commands
LAND_AT_END = True          # land after the last waypoint (mission can override)

# Waypoints outside this box are rejected (safety)
GEOFENCE = {
    "x": (-600, 600),
    "y": (-600, 600),
    "z": (40, 250),
}

# ---------------------------------------------------------------------------
# Downward camera (downvision)  -->  CALIBRATE THESE, see README
# ---------------------------------------------------------------------------
FRAME_IS_RGB = True         # djitellopy >= 2.4 delivers RGB frames
DOWNCAM_CROP = None         # (x, y, w, h) of the useful image area, or None
AUTO_CROP = True            # auto-detect black borders around the bottom image
CAM_HFOV_DEG = 60.0         # horizontal field of view of the bottom camera
CAM_FORWARD_SIGN = 1        # 1: top of image = drone forward, -1: flipped
CAM_LEFT_SIGN = 1           # 1: left of image = drone left,   -1: flipped
CAM_OFFSET_CM = (0.0, 0.0)  # camera position relative to drone centre (fwd, left)
FRAME_LATENCY_S = 0.2       # video delay; puddle position uses pose at t - latency

# ---------------------------------------------------------------------------
# Puddle detection
# ---------------------------------------------------------------------------
DETECT_HZ = 10              # detection runs (max) this many times per second
MIN_DETECT_HEIGHT = 30      # cm, no detection below this height (takeoff/landing)
DETECTOR_BACKEND = "threshold"  # "threshold" (no training), "yolo" or "roboflow"

# Trained object detection model (DETECTOR_BACKEND = "yolo" or "roboflow")
MODEL_PATH = "models/puddles.pt"      # yolo: .pt or .onnx from Ultralytics
ROBOFLOW_MODEL_ID = "puddles/1"       # roboflow: "<project>/<version>"
ROBOFLOW_API_KEY = None               # or set the ROBOFLOW_API_KEY environment variable
MODEL_CONFIDENCE = 0.5                # ignore boxes below this confidence
MODEL_IMGSZ = 320                     # yolo input size (same as used for training)
MODEL_CLASSES = None                  # e.g. ["puddle"]; None = accept every class
MODEL_BORDER_PX = 2                   # box this close to the edge = cut-off puddle

# Settings below are for the "threshold" backend (MIN_AREA_PX and
# REJECT_BORDER_BLOBS are used by every backend)
PUDDLE_MODE = "dark"        # "dark", "bright" (reflections) or "both"
BLUR_KERNEL = 7             # Gaussian blur kernel (odd number)
DARK_OFFSET = 30            # pixel must be this much darker than the floor median
BRIGHT_OFFSET = 60          # pixel must be this much brighter (mode bright/both)
MIN_AREA_PX = 250           # ignore blobs smaller than this (pixels)
MAX_AREA_FRAC = 0.7         # ignore blobs covering more than this part of the image
MIN_SOLIDITY = 0.5          # area / convex-hull area, removes thin lines/cracks
BORDER_MARGIN_PX = 4        # ignore this many pixels at the image border
REJECT_BORDER_BLOBS = True  # skip puddles cut off by the image edge (wrong centre)

# Tracking: merge repeated detections of the same puddle
MERGE_RADIUS_CM = 40        # detections closer than this belong to the same puddle
MIN_HITS = 3                # seen this many times before reported to the Jetson

# ---------------------------------------------------------------------------
# Recording downward frames for the dataset (--record DIR or key R / 'record')
# ---------------------------------------------------------------------------
RECORD_DIR = "dataset"      # default folder when recording is toggled on
RECORD_HZ = 2               # images per second (don't make near-identical images)
RECORD_ONLY_AIRBORNE = False  # True: only save while flying
