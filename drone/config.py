"""
Central configuration for the autonomous Tello EDU mission.

All distances are in centimetres, angles in degrees, unless stated otherwise.

Coordinate frame used everywhere (the "mission frame"):
    x = forward (the direction of the drone's nose / front camera at takeoff)
    y = left
    z = height above the floor
    origin = where the drone stands at takeoff
With RESET_FRAME_ON_TAKEOFF the frame is set again at every takeoff from the
ground (unless set_pose was sent just before), so the coordinates of a flight
always start at the drone itself, whichever way it was put down.
The Jetson converts real-world coordinates to this frame before sending them.
"""

# ---------------------------------------------------------------------------
# Network link with the Jetson (JSON over UDP, one message per datagram)
# ---------------------------------------------------------------------------
LISTEN_HOST = "127.0.0.1"   # loopback by default; set to 0.0.0.0 only if required
LISTEN_PORT = 9000          # Jetson sends commands to this port
JETSON_HOST = None          # None = reply to whoever sent the last command;
                            # an IP = only accept commands from that address
JETSON_PORT = 9001          # Jetson listens on this port for status/events
STATUS_INTERVAL = 1.0       # seconds between status messages

# ---------------------------------------------------------------------------
# Flight
# ---------------------------------------------------------------------------
DEFAULT_SPEED = 30          # cm/s for 'go' commands (10-100)
MAX_STEP_CM = 100           # long segments are split in steps of max this length
                            # (smaller = more corrections + faster abort)
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
RESET_FRAME_ON_TAKEOFF = True  # new mission frame at every takeoff (see the top of this file)
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
HELIPAD_MIN_AREA_PX = 60    # smallest H blob (pixels) that counts (about 10 x 10 px)
HELIPAD_SIZE_RANGE_CM = (8, 60)  # an H is between these sizes (only checked with a height)
HELIPAD_ADAPTIVE = True     # also threshold against the local brightness (H seen from higher up)
HELIPAD_ADAPTIVE_BLOCK = 41  # pixels, neighbourhood for that local brightness
HELIPAD_ADAPTIVE_C = 6       # pixel must be this much darker/brighter than its neighbourhood
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
FRAME_LATENCY_S = 0.2       # video delay; H position uses the pose at t - latency
DETECT_HZ = 10              # downward camera processed (H, recording, GUI view) per second

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
