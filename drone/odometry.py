"""
Odometry: measure where the drone really is, instead of only adding up commands.

* Heading: the IMU yaw of the Tello (state packets, no command needed).
* Height: the time-of-flight sensor under the drone.
* Horizontal movement: visual odometry on the downward camera. The shift of the
  floor texture between two frames (phase correlation) times the ground size of
  a pixel gives the real displacement, also while the drone "hovers" and slowly
  drifts or gets pushed by wind.

When the floor has too little texture (or the image is blurred) the odometry is
"lost" and the pose falls back to the commanded moves, like before.
"""

import collections
import math
import os
import threading
import time

import cv2
import numpy as np

from . import config as cfg
from .downcam import cam_to_body, cm_per_pixel
from .registration import recover_translation


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class Odometry:
    def __init__(self, tello, pose, frame_reader, cam, is_airborne, downcam_on=lambda: True):
        self.tello = tello
        self.pose = pose
        self.frame_reader = frame_reader
        self.cam = cam                # DownCam: grey conversion + crop
        self.is_airborne = is_airborne
        self.downcam_on = downcam_on  # False = the video is the front camera: no odometry
        self.running = True
        self._window = None
        self._control_lock = threading.RLock()
        self._flip_active = False
        self._flip_ref = None
        self._flip_capture = None
        self._reacquire = False
        self._resume_at = 0.0
        self._sensors = collections.deque(maxlen=80)
        self._good_frames = collections.deque(maxlen=6)
        self._recovery_diag = {}
        self._recovery_image = None
        self._recovery_candidate = None
        self._anchor_tracking = False
        self._return_active = False
        self._last_recovery_attempt = 0.0
        self._last_feature_attempt = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def capture_flip_reference(self):
        """Save the latest measured floor reference with the click coordinates.

        A prior loss of the flight origin does not invalidate a fresh local
        floor reference. No image decoding or SDK request is done in this call.
        """
        with self._control_lock:
            snapshot = self.pose.snapshot()
            reference = None
            now = time.time()
            recent = [entry for entry in self._good_frames if 0 <= now - entry[0] <= 0.35]
            if (not self._flip_active and not self._reacquire and recent
                    and not self.pose.vo_disabled
                    and self.downcam_on() and self.level()):
                # Only frames that actually measured a displacement belong here.
                # _prev may have been replaced by an unmatchable/blurred frame.
                frames = [dict(entry[1], img=entry[1]["img"].copy()) for entry in recent]
                reference = {"frame": frames[-1], "frames": frames, "world": self._world,
                             "pose": snapshot["pose"], "global_valid": snapshot["position_valid"],
                             "age_s": now - recent[-1][0], "quality": recent[-1][2]}
            return {"snapshot": snapshot, "reference": reference}

    def refresh_flip_reference(self, reference):
        """Add the settled view without changing the click's coordinate anchor."""
        with self._control_lock:
            if not reference or "frames" not in reference:
                return reference
            captured = self.capture_flip_reference()
            fresh = captured["reference"]
            if fresh is None or not captured["snapshot"]["measured"]:
                return reference
            # No frame may bridge a loss of continuity during the queue wait.
            if not (reference["global_valid"] and fresh["global_valid"]):
                return reference
            return dict(reference, frames=reference["frames"][-2:] + fresh["frames"][-6:],
                        before_sdk_frame=fresh["frame"])

    def begin_flip(self, reference=None):
        with self._control_lock:
            self._flip_capture = reference
            self._flip_ref = reference["frame"] if reference is not None else (self._prev if self.pose.measured else None)
            self._flip_valid = reference["global_valid"] if reference is not None else self.pose.position_valid
            self._flip_active = True
            self._anchor_tracking = False
            self._return_active = True
            self._last_recovery_attempt = 0.0
            self._recovery_candidate = None
            self._recovery_image = None
            self._recovery_diag = {"state": "waiting_sdk", "attempts": 0,
                                   "reference_saved": self._flip_ref is not None}
            self.pose.measured = False
            self.pose.position_valid = False
            self.pose.local_reference_valid = False

    def end_flip(self):
        with self._control_lock:
            self._flip_active = False
            self._reacquire = True
            self._recovery_diag["state"] = "waiting_video"
            # Video packets can still contain the rotation after SDK 'ok'.
            self._resume_at = time.time() + cfg.FRAME_LATENCY_S

    def finish_flip_recovery(self):
        with self._control_lock:
            # The motor return has a deadline; the saved visual anchor does not.
            # A later match may restore coordinates without restarting flight.
            self._return_active = False
            if self._anchor_tracking and not (self.pose.measured and self.pose.local_reference_valid):
                self._resume_anchor_search()
            else:
                self._anchor_tracking = False

    def _resume_anchor_search(self):
        self.pose.measured = False
        self.pose.position_valid = False
        self.pose.local_reference_valid = False
        self._anchor_tracking = False
        self._reacquire = True
        self._recovery_candidate = None
        self._key = self._prev = None

    def relocalize_from_pad(self, dx, dy):
        with self._control_lock:
            if self._flip_active or self._reacquire or self._anchor_tracking:
                return False
            return self.pose.relocalize_xy(dx, dy)

    def reset_tracking(self):
        with self._control_lock:
            self._key = self._prev = self._prev_frame = None
            self._world = (0.0, 0.0)
            self._last_good = 0.0
            self._last_feature_attempt = 0.0
            self._sensors.clear()
            self._good_frames.clear()
            self._flip_active = self._reacquire = False
            self._anchor_tracking = False
            self._return_active = False
            self._last_recovery_attempt = 0.0
            self._recovery_candidate = None
            self._flip_ref = None
            self._flip_capture = None
            self.pose.local_reference_valid = False
            self._resume_at = 0.0

    def level(self):
        try:
            state = self.tello.get_current_state()
            return max(abs(float(state[axis])) for axis in ("pitch", "roll")) <= cfg.VO_MAX_TILT_DEG
        except (KeyError, TypeError, ValueError, AttributeError):
            return False

    def _gray(self, frame):
        gray = self.cam.cropped(self.cam.to_gray(frame))
        if gray.shape[1] > 320:  # keep it fast; the scale follows the image width
            f = 320 / gray.shape[1]
            gray = cv2.resize(gray, (320, int(round(gray.shape[0] * f))),
                              interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32)

    @staticmethod
    def _normalisation_matrix(ref, shape, height=None, yaw=None):
        angle = 0 if yaw is None else wrap_deg(yaw - ref["yaw"])
        scale = 1 if height is None else height / ref["height"]
        h, w = shape
        return cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)

    @staticmethod
    def _normalise(ref, g, height=None, yaw=None):
        angle = 0 if yaw is None else wrap_deg(yaw - ref["yaw"])
        scale = 1 if height is None else height / ref["height"]
        if abs(scale - 1) > 0.005 or abs(angle) > 0.1:
            h, w = g.shape
            transform = Odometry._normalisation_matrix(ref, g.shape, height, yaw)
            return cv2.warpAffine(g, transform, (w, h))
        return g

    def _compare(self, ref, g, height=None, yaw=None):
        if self._window is None or self._window.shape != g.shape:
            self._window = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        if ref["img"].shape != g.shape:
            return (0.0, 0.0), 0.0
        # A climb scales the floor image. Undo that scale before estimating
        # translation, otherwise a vertical movement becomes false xy drift.
        g = self._normalise(ref, g, height, yaw)
        shift, quality = cv2.phaseCorrelate(ref["img"].copy(), g.copy(), self._window)
        sx, sy = shift
        h, w = g.shape
        if not all(math.isfinite(v) for v in (sx, sy, quality)) or abs(sx) > w / 3 or abs(sy) > h / 3:
            return shift, 0.0
        # The phase response alone can accept a blurred, unrelated frame.
        aligned = cv2.warpAffine(g, np.float32([[1, 0, -sx], [0, 1, -sy]]), (w, h))
        margin = int(max(abs(sx), abs(sy))) + 8
        a = ref["img"][margin:h-margin, margin:w-margin]
        b = aligned[margin:h-margin, margin:w-margin]
        if a.size < 100 or a.std() < 2 or b.std() < 2:
            return shift, 0.0
        correlation = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        return shift, quality if correlation >= cfg.VO_MIN_CORRELATION else 0.0

    @staticmethod
    def _features(g):
        sift = cv2.SIFT_create(nfeatures=1600, contrastThreshold=0.01)
        return sift.detectAndCompute(np.clip(g, 0, 255).astype(np.uint8), None)

    def _feature_shift(self, ref, g, height, yaw, current_features=None):
        """Plane registration for larger offsets or residual perspective changes.

        Features alone never authorize motion: require mutual descriptor matches,
        RANSAC inliers spread across the floor, plausible geometry and image agreement.
        """
        # SIFT already handles rotation and scale. Warping before detection
        # interpolates away small features and clips the very overlap needed
        # after a large flip displacement. Apply sensor normalisation to the
        # fitted geometry instead, retaining the original pixels/descriptors.
        if ref["img"].shape != g.shape:
            return None, {"reason": "image_shape_changed"}
        if "features" not in ref:
            ref["features"] = self._features(ref["img"])
        first, desc1 = ref["features"]
        second, desc2 = self._features(g) if current_features is None else current_features
        info = {"reference_features": len(first), "current_features": len(second)}
        if desc1 is None or desc2 is None or min(len(first), len(second)) < 12:
            return None, dict(info, reason="not_enough_floor_features")
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        reverse = {m.queryIdx: m.trainIdx for m in matcher.match(desc2, desc1)}
        matches = [pair[0] for pair in matcher.knnMatch(desc1, desc2, k=2)
                   if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
                   and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
        info["matches"] = len(matches)
        if len(matches) < 12:
            return None, dict(info, reason="not_enough_matching_features")
        source = np.float32([first[m.queryIdx].pt for m in matches])
        target = np.float32([second[m.trainIdx].pt for m in matches])
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 2.5,
                                          maxIters=1500, confidence=0.995)
        if matrix is None or mask is None:
            return None, dict(info, reason="no_consistent_plane")
        keep = mask.ravel().astype(bool)
        info["inliers"] = int(keep.sum())
        h, w = g.shape
        if keep.sum() < 12 or keep.mean() < 0.55:
            return None, dict(info, reason="features_not_consistent_or_spread")
        spread = np.ptp(source[keep], axis=0)
        if spread.prod() < 0.04 * w * h:
            return None, dict(info, reason="features_not_consistent_or_spread")
        try:
            inverse = np.linalg.inv(matrix)
            normalisation = np.vstack((self._normalisation_matrix(ref, g.shape, height, yaw),
                                       (0, 0, 1)))
            normalised_inverse = np.linalg.inv(normalisation @ matrix)
        except np.linalg.LinAlgError:
            return None, dict(info, reason="singular_plane")
        centre = np.float32([[(w - 1) / 2, (h - 1) / 2]])
        probes = np.concatenate([centre, centre + (1, 0), centre + (0, 1)]).astype(np.float32)
        projected = cv2.perspectiveTransform(probes[None], normalised_inverse)[0]
        jacobian = (projected[1:] - projected[0]).T
        determinant = np.linalg.det(jacobian)
        angle = math.degrees(math.atan2(jacobian[1, 0], jacobian[0, 0]))
        if not 0.5 < determinant < 2 or abs(angle) > 15:
            return None, dict(info, reason="plane_disagrees_with_imu_or_height")
        info.update(residual_rotation_deg=float(angle), residual_scale=float(math.sqrt(determinant)))
        aligned = cv2.warpPerspective(g, inverse, (w, h))
        valid = cv2.warpPerspective(np.ones((h, w), np.uint8), inverse, (w, h)) > 0
        valid[:8] = valid[-8:] = False
        valid[:, :8] = valid[:, -8:] = False
        if valid.sum() < 0.2 * w * h or min(aligned[valid].std(), ref["img"][valid].std()) < 2:
            return None, dict(info, reason="not_enough_textured_overlap")
        correlation = float(np.corrcoef(aligned[valid], ref["img"][valid])[0, 1])
        info["correlation"] = correlation
        if not math.isfinite(correlation) or correlation < cfg.VO_MIN_CORRELATION:
            return None, dict(info, reason="registered_images_disagree")
        shift = centre[0] - projected[0]
        return tuple(float(v) for v in shift), dict(info, reason="matched_features")

    def recovery_status(self):
        with self._control_lock:
            return dict(self._recovery_diag)

    def _recover_position(self, g, height, yaw):
        capture = self._flip_capture
        refs = capture.get("frames", [self._flip_ref]) if capture else [self._flip_ref]
        info = {"state": "matching", "attempts": self._recovery_diag.get("attempts", 0) + 1,
                "height_cm": height, "yaw_deg": yaw}
        possible = [ref for ref in reversed(refs) if ref is not None
                    and 0.45 < height / ref["height"] < 2.2
                    and abs(wrap_deg(yaw - ref["yaw"])) <= 35]
        for ref in possible:
            shift, quality = self._compare(ref, g, height, yaw)
            info.update(phase_quality=float(quality), yaw_change_deg=wrap_deg(yaw - ref["yaw"]),
                        height_ratio=height / ref["height"])
            if quality >= max(0.2, cfg.VO_MIN_RESPONSE):
                new = self._position(ref, shift, g.shape[1], ref["height"])
                self._recovery_diag = dict(info, state="matched_translation")
                return new, quality
        # A flip can move the scene beyond the small-step VO window. Validate
        # the genuinely shared pixels, without a symmetric crop that erases
        # all overlap when one axis has shifted by more than a third-frame.
        for ref in possible:
            shift, quality, overlap = recover_translation(
                ref["img"], g, height / ref["height"], wrap_deg(yaw - ref["yaw"]))
            if shift is not None:
                new = self._position(ref, shift, g.shape[1], ref["height"])
                if math.dist(new, ref["world"]) > 100:
                    continue
                self._recovery_diag = dict(info, **overlap, state="matched_overlap_translation")
                return new, quality
        # All saved pre-click frames share this current image; extract once.
        current_features = self._features(g) if possible else None
        for ref in possible:
            shift, features = self._feature_shift(ref, g, height, yaw, current_features)
            info.update(features)
            if shift is not None:
                new = self._position(ref, shift, g.shape[1], ref["height"])
                if math.dist(new, ref["world"]) > 100:
                    info["reason"] = "displacement_above_100_cm"
                    continue
                self._recovery_diag = dict(info, state="matched_features")
                return new, 0.5
        self._recovery_diag = dict(info, state="searching",
                                   reason=info.get("reason", "height_or_heading_outside_registration_range"
                                                   if not possible else "no_image_agreement"))
        return None, 0.0

    def save_flip_diagnostics(self, directory):
        with self._control_lock:
            info = dict(self._recovery_diag)
            for name, gray in (("reference", self._flip_ref["img"] if self._flip_ref else None),
                               ("before_sdk", self._flip_capture["before_sdk_frame"]["img"]
                                if self._flip_capture and "before_sdk_frame" in self._flip_capture else None),
                               ("after", self._recovery_image)):
                if gray is not None:
                    filename = f"last_flip_{name}.png"
                    if cv2.imwrite(os.path.join(directory, filename), np.clip(gray, 0, 255).astype(np.uint8)):
                        info[name + "_image"] = filename
            return info

    @staticmethod
    def _position(ref, shift, width, height):
        """Odometry position for a frame shifted `shift` pixels from reference `ref`."""
        sx, sy = shift
        # floor moves down in the image = drone moves forward, etc.
        s = cm_per_pixel(width, (height + ref["height"]) / 2)
        fwd, left = cam_to_body(sy * s * cfg.CAM_FORWARD_SIGN, sx * s * cfg.CAM_LEFT_SIGN)
        ky = math.radians(ref["yaw"])
        return (ref["world"][0] + fwd * math.cos(ky) - left * math.sin(ky),
                ref["world"][1] + fwd * math.sin(ky) + left * math.cos(ky))

    def _height(self):
        try:
            h = self.tello.get_distance_tof()
        except Exception:
            return None
        return float(h) if 10 < h < 1000 else None

    def _loop(self):
        """
        Keyframe visual odometry: every frame is compared with a reference frame
        (keyframe), not with the previous frame, so the small sub-pixel error of
        each comparison does not add up while hovering or flying slowly. A new
        keyframe is taken when the view has shifted far enough, the drone turned
        or the height changed.
        """
        period = 1.0 / cfg.VO_HZ
        self._key = self._prev = None   # dict(img, world, yaw, height)
        self._prev_frame = None
        self._world = (0.0, 0.0)        # odometry position (only differences are used)
        self._last_good = 0.0
        last_t = time.time()
        error = None
        while self.running:
            t0 = time.time()
            dt, last_t = t0 - last_t, t0
            try:
                self._step(t0, dt)
            except Exception as e:
                # never let this thread die: the pose would keep saying "measured"
                if str(e) != error:
                    error = str(e)
                    print(f"⚠️  Fout in de odometrie: {e!r}")
                self._key = self._prev = self._prev_frame = None
                self.pose.position_valid = False
                self.pose.local_reference_valid = False
                self.pose.step(dt, t0, None, None, measured=False)
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _step(self, t0, dt):
        with self._control_lock:
            self._step_locked(t0, dt)

    def _step_locked(self, t0, dt):
        if getattr(self, "_anchor_tracking", False) and t0 - self._last_good > cfg.VO_LOST_S:
            # A frozen stream, unavailable sensors or a tilted camera must also
            # break continuity, even when there is no new image to reject.
            self._resume_anchor_search()
        level = self.level()
        if self._flip_active or t0 < self._resume_at or not level:
            if self._reacquire:
                self._recovery_diag["state"] = "waiting_video" if level else "waiting_level_attitude"
            if not self._flip_active and level and hasattr(self, "_sensors"):
                self.pose.update_imu(self.tello.get_yaw())
                self._sensors.append((t0, self._height(), self.pose.yaw))
            self.pose.step(dt, t0, measured=False)
            return
        key, prev, prev_frame = self._key, self._prev, self._prev_frame
        world, last_good = self._world, self._last_good
        try:
            self.pose.update_imu(self.tello.get_yaw())
        except Exception:
            pass
        yaw = self.pose.yaw

        vo = None
        height = None
        quality = 0.0
        airborne = self.is_airborne()
        usable = False
        if airborne:
            height = self._height()
            actual_height = height
            # The decoded picture is older than the latest ToF/yaw packet.
            # Match its scale to the sensor sample at capture time, especially
            # during takeoff where the height changes quickly.
            sensors = getattr(self, "_sensors", None)
            if sensors is not None:
                sensors.append((t0, height, yaw))
                capture_time = t0 - cfg.FRAME_LATENCY_S
                sample = next((s for s in reversed(sensors) if s[0] <= capture_time), None)
                if sample is None:
                    height = None
                else:
                    height, yaw = sample[1:]
            frame = self.frame_reader.frame
            usable = (cfg.VO_ENABLED and not self.pose.vo_disabled and height is not None
                      and height >= cfg.VO_MIN_HEIGHT and frame is not None
                      and self.downcam_on())
            if usable and frame is not prev_frame:  # only new frames carry information
                prev_frame = frame
                g = self._gray(frame)
                if self._reacquire or getattr(self, "_anchor_tracking", False):
                    self._recovery_image = g.copy()
                    matched = False
                    anchored = getattr(self, "_anchor_tracking", False)
                    if (self._reacquire and t0 - self._resume_at > cfg.FLIP_TRACK_TIMEOUT_S
                            and t0 - self._last_recovery_attempt < 0.25):
                        # Continue relocalizing after the flight command ends,
                        # at 4 Hz so two fresh matches can still confirm within
                        # 0.35 s. Never seed a new origin from an unknown image.
                        self._prev_frame = frame
                        self.pose.step(dt, t0, height=height, measured=False)
                        return
                    self._last_recovery_attempt = t0
                    new, quality = self._recover_position(g, height, yaw)
                    if new is not None:
                        candidate = self._recovery_candidate
                        count = (candidate[2] + 1 if candidate is not None
                                 and t0 - candidate[0] < 0.35
                                 and math.dist(new, candidate[1]) < 5 else 1)
                        self._recovery_candidate = (t0, new, count)
                        self._recovery_diag["consistent_frames"] = count
                        self._recovery_diag["method"] = self._recovery_diag["state"]
                        self._recovery_diag["state"] = "reference_recovered" if count >= 2 else "confirming_match"
                        if count >= 2 or anchored:
                            capture = self._flip_capture
                            if capture is not None:
                                # Anchor to the click, even if the worker started
                                # later or global command estimates changed meanwhile.
                                current = self.pose.get()
                                delta = tuple(capture["pose"][i] + new[i] - capture["world"][i]
                                              - current[i] for i in (0, 1))
                            else:
                                delta = (new[0] - world[0], new[1] - world[1])
                            self.pose.step(dt, t0, delta,
                                           height, measured=True, quality=quality)
                            self.pose.position_valid = self._flip_valid
                            self.pose.local_reference_valid = True
                            self._world, self._last_good = new, t0
                            self._key = self._prev = dict(img=g, yaw=yaw, height=height, world=new)
                            self._good_frames.append((t0, self._prev, quality))
                            matched = True
                    if matched:
                        self._reacquire = False
                        self._anchor_tracking = self._return_active
                        if not self._return_active:
                            print("📍 Positie opnieuw herkend via opgeslagen camerabeeld; "
                                  "geen automatische terugkeermanoeuvre gestart")
                    self._prev_frame = frame
                    if not matched:
                        self.pose.step(dt, t0, height=height, measured=False)
                    return
                current = {"img": g, "yaw": yaw, "height": height}
                if (key is None or key["img"].shape != g.shape
                        or abs(wrap_deg(yaw - key["yaw"])) > cfg.VO_MAX_YAW_STEP
                        or abs(height / key["height"] - 1) > 0.12):
                    # turned or climbed: the old keyframe can't be compared anymore.
                    # While climbing/descending (takeoff, landing, over a box) the
                    # previous frame is still close enough: measure that last step
                    # first, so the drift during a height change is not lost.
                    if (prev is not None and prev["img"].shape == g.shape
                            and abs(wrap_deg(yaw - prev["yaw"])) <= cfg.VO_MAX_YAW_STEP
                            and abs(height / prev["height"] - 1) <= 0.12):
                        shift, quality = self._compare(prev, g, height)
                        if quality >= cfg.VO_MIN_RESPONSE:
                            new = self._position(prev, shift, g.shape[1], prev["height"])
                            vo = (new[0] - world[0], new[1] - world[1])
                            world = new
                            last_good = t0
                    key = prev = dict(current, world=world)
                else:
                    # 1) against the keyframe, 2) if that fails against the previous frame
                    for ref in (key, prev):
                        shift, quality = self._compare(ref, g, height)
                        if quality >= cfg.VO_MIN_RESPONSE:
                            break
                    # Phase correlation rejects large translations and small
                    # perspective changes. Try geometric registration before
                    # dropping the displacement and restarting at the old xy.
                    # Limit its rate so ordinary camera tracking stays responsive.
                    if quality < cfg.VO_MIN_RESPONSE and t0 - getattr(self, "_last_feature_attempt", 0) >= 0.2:
                        self._last_feature_attempt = t0
                        features = self._features(g)
                        for candidate in (key, prev) if key is not prev else (key,):
                            feature_shift, _ = self._feature_shift(candidate, g, height, yaw, features)
                            if feature_shift is not None:
                                ref, shift, quality = candidate, feature_shift, 0.5
                                break
                    if quality >= cfg.VO_MIN_RESPONSE:
                        new = self._position(ref, shift, g.shape[1], ref["height"])
                        vo = (new[0] - world[0], new[1] - world[1])
                        world = new
                        last_good = t0
                        prev = dict(current, world=world)
                        if (ref is not key or quality < cfg.VO_KEY_QUALITY
                                or math.hypot(*shift) > cfg.VO_KEY_SHIFT_PX):
                            key = prev
                    elif t0 - last_good > cfg.VO_LOST_S:
                        # lost (no texture/blur): restart from this frame
                        self.pose.position_valid = False
                        self.pose.local_reference_valid = False
                        key = prev = dict(current, world=world)
            elif not usable:
                if self._reacquire:
                    self._recovery_diag.update(state="waiting_sensors", height_cm=height,
                        reason="downward_camera_disabled" if not self.downcam_on() else
                               "invalid_height" if height is None or height < cfg.VO_MIN_HEIGHT else
                               "no_video_frame" if frame is None else "odometry_disabled")
                key = None
                if not self.downcam_on():
                    prev = prev_frame = None   # front camera: these frames don't compare
                    self.pose.position_valid = False
                    self.pose.local_reference_valid = False
                    self._good_frames.clear()
                    if getattr(self, "_anchor_tracking", False):
                        self._resume_anchor_search()
        else:
            # on the ground: forget the previous frame too, otherwise the first frame
            # of the next flight is compared with the landing of this one (false jump)
            key = prev = prev_frame = None
            self._last_good = last_good = 0.0
            if hasattr(self, "_sensors"):
                self._sensors.clear()
            self._good_frames.clear()

        vo_ok = (cfg.VO_ENABLED and not self.pose.vo_disabled and airborne
                 and usable and t0 - last_good < cfg.VO_LOST_S)
        if vo is not None and math.hypot(*vo) > cfg.VO_MAX_SPEED_CMS * max(dt, t0 - self._last_good) + 3:
            vo = None
            vo_ok = False
            self.pose.position_valid = False
            self.pose.local_reference_valid = False
            key = prev = None
            world = self._world
        self.pose.step(dt, t0, vo, actual_height if airborne else None, measured=vo_ok,
                       quality=quality)
        if vo_ok and vo is not None:
            self._good_frames.append((t0, dict(img=g, yaw=yaw, height=height, world=world), quality))
        self._key, self._prev, self._prev_frame = key, prev, prev_frame
        self._world, self._last_good = world, last_good

    def stop(self):
        self.running = False
