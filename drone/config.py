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
LISTEN_HOST = "127.0.0.1"   # loopback by default; set to 0.0.0.0 only if required
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
MIN_BATTERY = 5             # refuse a mission below this battery %
RESPONSE_TIMEOUT = 20       # s, djitellopy wait time for 'ok' (long go commands)
KEEPALIVE_INTERVAL = 5      # s, Tello lands by itself after 15 s without commands
LAND_AT_END = True          # land after the last waypoint (mission can override)

# Navigation mode (the GUI can change it per mission)
#   "go": straight 'go' moves between waypoints (the drone stops at every point);
#         drift and rotation are corrected before each move
#   "rc": smooth flight with continuous steering (no stops), needs visual odometry
NAV_MODE = "go"
FINE_POSITION = "auto"      # precise positioning at waypoints: "off", "last", "all" or
                            # "auto" (= "last": only the end point, so the drone doesn't
                            # stop and shuffle around at every waypoint)
Z_DEADBAND_CM = 25          # height errors smaller than this are NOT corrected: a small step
                            # in the floor changes the ToF height, the drone should not bob
                            # up and down for it (0 = always correct the height)
FINE_TOL_CM = 10            # precise positioning: stop when closer than this
FINE_TIMEOUT_S = 4          # give up precise positioning after this time

# The Tello slides sideways while it takes off and lands. With working odometry:
TAKEOFF_RECENTER = True     # after takeoff, move back above the takeoff spot
PRECISE_LAND = True         # mission end: descend slowly while holding the spot, then land
LAND_HOVER_CM = 30          # ... down to this height, the last bit is the normal 'land'
LAND_DESCENT_CMS = 25       # descent speed (cm/s) during the precise landing

# Helipad: land in the middle of an "H" landing pad (drone/helipad.py)
HELIPAD_LAND = True         # mission end: look for an H near the last waypoint and land on it
HELIPAD_RADIUS_CM = 80      # only an H within this distance of the last waypoint counts
                            # (the camera sees about +-45 cm around the drone at 80 cm, +-85 at 150)
HELIPAD_SEARCH_S = 2.0      # hover this long at the end point looking for the H
HELIPAD_SEARCH_HEIGHT_CM = 150  # not found: climb to this height (sees more floor) and look again
HELIPAD_CLIMB_CMS = 30      # climb speed for that
HELIPAD_CENTER_TOL_CM = 6   # only descend while the H is this close below the drone centre
HELIPAD_FINAL_CM = 40       # centred at this height -> normal 'land' for the last bit
HELIPAD_FINAL_SIZE = 0.5    # ... or as soon as the H is this part of the image width
HELIPAD_DESCENT_CMS = 20    # descent speed above the H
HELIPAD_TIMEOUT_S = 25      # give up (normal landing) after this time
HELIPAD_MIN_AREA_PX = 300   # smallest H blob (pixels) that counts
HELIPAD_MAX_SCORE = 0.4     # difference with an ideal H (0 = perfect), higher = looser

# Heading hold: keep the nose in the start direction (IMU yaw)
YAW_HOLD = True
YAW_TOL_DEG = 6             # 'go' mode: rotate back when the heading is off by more than this
YAW_SIGN = -1               # Tello yaw is clockwise-positive, mission yaw counter-clockwise
                            # CHECK: rotate the drone left by hand -> GUI yaw must go UP

# rc-mode controller
RC_CMS_PER_UNIT = 1.0       # approx. speed (cm/s) per rc unit (rc goes from -100 to 100)
RC_LOOKAHEAD_CM = 40        # follow a point this far ahead on the path (pure pursuit);
                            # further = smoother, fewer corrections
PATH_TOL_CM = 15            # rc mode: up to this far beside the path is fine, the drone just
                            # flies parallel to it; only the part beyond this is corrected
RC_PASS_CM = 12             # waypoint counts as passed within this distance
RC_GAIN = 1.2               # 1/s, speed towards the path per cm of error
RC_YAW_GAIN = 1.5           # rc yaw units per degree heading error
RC_HZ = 15                  # control loop rate
RC_MIN_UNITS = 8            # smallest rc value that still moves the drone
RC_SETTLE_S = 1.0           # wait this long after rc steering before a 'go' (otherwise the
                            # Tello answers "error Not joystick")

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
CAM_ROTATE_DEG = 270          # camera image turned 0/90/180/270 deg relative to the drone.
                            # Detected automatically on the first move of a flight
                            # (log: "Camerabeeld is ... gedraaid"); put that value here.
CAM_OFFSET_CM = (0.0, 0.0)  # camera position relative to drone centre (fwd, left)
FRAME_LATENCY_S = 0.2       # video delay; puddle position uses pose at t - latency

# ---------------------------------------------------------------------------
# Visual odometry: measure the real movement from the downward camera
# (shift of the floor between frames). Corrects drift while hovering and wind.
# Needs some texture on the floor; falls back to the commanded moves if not.
# ---------------------------------------------------------------------------
VO_ENABLED = True
VO_HZ = 20                  # frames per second used for odometry
VO_MIN_RESPONSE = 0.08      # phase correlation quality, lower = no texture/blur
VO_MIN_HEIGHT = 20          # cm, no odometry below this height
VO_MAX_YAW_STEP = 1.5       # deg turned since the keyframe; more = take a new keyframe
VO_KEY_SHIFT_PX = 20        # new keyframe after this much image shift
VO_KEY_QUALITY = 0.3        # ... or when the match quality drops below this
VO_LOST_S = 0.7             # s without good frames = odometry lost

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
# Telemetry (sensor graphs in the GUI) and terrain analysis
# Terrain: ground elevation = barometer altitude - ToF distance to the ground
# ---------------------------------------------------------------------------
TELEMETRY_HZ = 10           # sensor samples per second (the Tello sends ~10 state packets/s)
TELEMETRY_HISTORY_S = 600   # keep this much history (graphs + CSV export)
TOF_MIN_CM = 10             # the ToF sensor reports 10 when it has no valid reading
TOF_MAX_CM = 800
TERRAIN_ALT_SOURCE = "baro"  # "baro" (barometer) or "h" (the Tello's own height estimate)
TERRAIN_ALT_TAU_S = 0.4     # s, smoothing of the (noisy) barometer altitude
TERRAIN_CELL_CM = 20        # grid cell size of the terrain map
TERRAIN_REF_SAMPLES = 15    # first samples in the air define the floor level (0 cm)
TERRAIN_MIN_SAMPLES = 3     # a cell needs this many samples to count in the analysis
TERRAIN_OBSTACLE_CM = 15    # higher than this above the floor = obstacle (lower = dip)
TERRAIN_PROFILE_MAX = 6000  # samples kept for the height profile along the track

# ---------------------------------------------------------------------------
# Recording downward frames for the dataset (--record DIR or key R / 'record')
# ---------------------------------------------------------------------------
RECORD_DIR = "dataset"      # default folder when recording is toggled on
RECORD_HZ = 2               # images per second (don't make near-identical images)
RECORD_ONLY_AIRBORNE = False  # True: only save while flying
