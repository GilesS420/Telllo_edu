import ast
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import calibrate_camera as cli
from drone.camera_calibration import (CalibrationError, infer_camera_transform,
                                      validate_capture_pair)


class CameraAxisInferenceTests(unittest.TestCase):
    @staticmethod
    def observations(rotation, left_sign, angle_error_deg=0):
        angle = math.radians(rotation)
        transform = np.array([[math.cos(angle), -math.sin(angle) * left_sign],
                              [math.sin(angle), math.cos(angle) * left_sign]])
        error = math.radians(angle_error_deg)
        noise = np.array([[math.cos(error), -math.sin(error)],
                          [math.sin(error), math.cos(error)]])
        raw = transform.T @ noise @ np.diag([20, 20])
        # raw=(shift_y, shift_x)*scale, scale=0.5 cm/pixel.
        return raw[::-1, 0] / 0.5, raw[::-1, 1] / 0.5

    def infer(self, forward, left, scale=0.5):
        return infer_camera_transform(forward, left, forward_cm_per_pixel=scale,
                                      left_cm_per_pixel=scale)

    def test_all_eight_signed_quarter_turns(self):
        for rotation in (0, 90, 180, 270):
            for left_sign in (-1, 1):
                with self.subTest(rotation=rotation, left_sign=left_sign):
                    fit = self.infer(*self.observations(rotation, left_sign))
                    self.assertEqual(fit["CAM_ROTATE_DEG"], rotation)
                    self.assertEqual(fit["CAM_FORWARD_SIGN"], 1)
                    self.assertEqual(fit["CAM_LEFT_SIGN"], left_sign)
                    self.assertAlmostEqual(fit["forward_alignment_cos"], 1)
                    self.assertAlmostEqual(fit["left_alignment_cos"], 1)

    def test_small_manual_angle_error_still_identifies_axes(self):
        for rotation in (0, 90, 180, 270):
            for sign in (-1, 1):
                with self.subTest(rotation=rotation, sign=sign):
                    fit = self.infer(*self.observations(rotation, sign, 8))
                    self.assertEqual((fit["CAM_ROTATE_DEG"], fit["CAM_LEFT_SIGN"]),
                                     (rotation, sign))

    def test_separate_height_scales_are_allowed(self):
        fit = infer_camera_transform((0, 40), (20, 0),
                                     forward_cm_per_pixel=0.5, left_cm_per_pixel=1)
        self.assertEqual(fit["CAM_ROTATE_DEG"], 0)

    def test_same_direction_and_nonorthogonal_moves_are_rejected(self):
        for left in ((0, 40), (0, -40), (25, 25), (30, 10)):
            with self.subTest(left=left), self.assertRaises(CalibrationError):
                self.infer((0, 40), left)

    def test_diagonal_orthogonal_moves_are_ambiguous(self):
        with self.assertRaisesRegex(CalibrationError, "ambigus"):
            self.infer(*self.observations(0, 1, 45))

    def test_wrong_axis_even_if_other_move_aligns_is_rejected(self):
        # Only one matching direction is insufficient; retain the >.92 check
        # even when the perpendicularity tolerance is independently relaxed.
        with self.assertRaises(CalibrationError):
            self.infer((0, 40), (20, 35))

    def test_immobile_tiny_and_excessive_moves_are_rejected(self):
        for forward, scale in (((0, 0), 0.5), ((0, 2), 5), ((0, 10), 0.5),
                               ((0, 100), 0.5)):
            with self.subTest(forward=forward, scale=scale), self.assertRaises(CalibrationError):
                self.infer(forward, (40, 0), scale)

    def test_invalid_shifts_and_scales_are_rejected(self):
        for forward, scale in (((0, float("nan")), 0.5), ((0, float("inf")), 0.5),
                               ((0,), 0.5), (None, 0.5), ((0, 40), 0),
                               ((0, 40), -0.5), ((0, 40), float("nan"))):
            with self.subTest(forward=forward, scale=scale), self.assertRaises(CalibrationError):
                self.infer(forward, (40, 0), scale)


class CaptureValidationTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(height_cm=80, yaw_deg=179, pitch_deg=0, roll_deg=0)

    def test_stable_level_samples_and_yaw_wrap_are_accepted(self):
        validate_capture_pair(self.reference, dict(self.reference, yaw_deg=-179))

    def test_missing_and_nonfinite_sensors_are_rejected(self):
        for sample in ({}, dict(self.reference, height_cm=None),
                       dict(self.reference, yaw_deg=float("inf")),
                       dict(self.reference, pitch_deg=float("nan"))):
            with self.subTest(sample=sample), self.assertRaises(CalibrationError):
                validate_capture_pair(self.reference, sample)

    def test_invalid_height_reference_or_current_is_rejected(self):
        for height in (0, 10, 59, 121, 8190):
            bad = dict(self.reference, height_cm=height)
            with self.subTest(height=height), self.assertRaises(CalibrationError):
                validate_capture_pair(bad, self.reference)
            with self.subTest(height=height), self.assertRaises(CalibrationError):
                validate_capture_pair(self.reference, bad)

    def test_tilt_yaw_and_scale_changes_are_rejected(self):
        for values in (dict(roll_deg=6), dict(pitch_deg=-6), dict(yaw_deg=170),
                       dict(height_cm=93), dict(height_cm=67)):
            with self.subTest(values=values), self.assertRaises(CalibrationError):
                validate_capture_pair(self.reference, dict(self.reference, **values))


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(24)
        self.image = cv2.GaussianBlur(rng.uniform(0, 255, (240, 320)).astype(np.float32),
                                      (0, 0), 0.8)
        self.reference = dict(img=self.image, height_cm=80, yaw_deg=0,
                              pitch_deg=0, roll_deg=0)

    def test_actual_image_translation_has_the_expected_sign(self):
        translated = cv2.warpAffine(self.image, np.float32([[1, 0, -25], [0, 1, 12]]),
                                    (320, 240))
        result = cli._measure_shift(self.reference, dict(self.reference, img=translated))
        np.testing.assert_allclose(result["shift_px"], (-25, 12), atol=0.3)
        self.assertEqual(result["registration"], "phase")

    def test_feature_registration_handles_a_larger_shift(self):
        translated = cv2.warpAffine(self.image, np.float32([[1, 0, -110], [0, 1, 0]]),
                                    (320, 240))
        result = cli._measure_shift(self.reference, dict(self.reference, img=translated))
        np.testing.assert_allclose(result["shift_px"], (-110, 0), atol=0.5)
        self.assertEqual(result["registration"], "features")

    def test_unrelated_and_textureless_images_are_rejected(self):
        for image in (np.zeros_like(self.image),
                      np.random.default_rng(81).uniform(0, 255, self.image.shape).astype(np.float32)):
            with self.subTest(texture=float(image.std())), self.assertRaises(CalibrationError):
                cli._measure_shift(self.reference, dict(self.reference, img=image))

    def test_bad_sensor_pair_is_rejected_before_registration(self):
        with patch.object(cli.Odometry, "_compare") as compare:
            with self.assertRaises(CalibrationError):
                cli._measure_shift(self.reference, dict(self.reference, yaw_deg=10))
            compare.assert_not_called()

    def test_feature_geometry_exposes_rotation_or_scale_missed_by_sensors(self):
        for rotation, scale in ((8, 1), (0, 1.2)):
            with self.subTest(rotation=rotation, scale=scale):
                matrix = cv2.getRotationMatrix2D((159.5, 119.5), rotation, scale)
                matrix[:, 2] += (0, 20)
                image = cv2.warpAffine(self.image, matrix, (320, 240))
                with self.assertRaises(CalibrationError):
                    cli._measure_shift(self.reference, dict(self.reference, img=image))

    def test_warmup_prepares_ten_fresh_frames_and_fixes_crop(self):
        from drone.downcam import DownCam

        frames = [np.pad(np.full((80, 100), 100, np.uint8), ((10, 10), (10, 10)))
                  for _ in range(10)]
        camera = DownCam()
        with patch.object(cli, "_new_frame", side_effect=frames), \
                patch.object(cli.cfg, "AUTO_CROP", True), \
                patch.object(cli.cfg, "DOWNCAM_CROP", None), \
                contextlib.redirect_stdout(io.StringIO()):
            cli._warmup_camera(object(), camera)
        self.assertEqual(camera.crop, (10, 10, 100, 80))


class NonFlyingCliTests(unittest.TestCase):
    class Reader:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class VideoOnlyTello:
        """Deliberately has no flight methods, end(), or motor-control API."""

        def __init__(self):
            self.calls = []
            self.reader = NonFlyingCliTests.Reader()

        def connect(self):
            self.calls.append("connect")

        def streamon(self):
            self.calls.append("streamon")

        def send_command_with_return(self, command):
            if command != "downvision 1":
                raise AssertionError(f"Unexpected SDK command: {command}")
            self.calls.append(command)
            return "ok"

        def get_frame_read(self):
            return self.reader

        def streamoff(self):
            self.calls.append("streamoff")

    def run_cli(self, tello, directory, measurements):
        with patch("drone.connection.ConfiguredTello", return_value=tello), \
                patch("builtins.input", return_value=""), \
                patch.object(cli.time, "sleep"), \
                patch.object(cli, "_warmup_camera"), \
                patch.object(cli, "_capture_movement", side_effect=measurements), \
                patch.object(cli, "WORKSPACE", Path(directory)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = cli.main([])
        return code, output.getvalue()

    def test_cli_only_uses_video_commands_and_writes_reviewable_proposal(self):
        tello = self.VideoOnlyTello()
        observations = [{"shift_px": [0, 40], "cm_per_pixel": 0.5},
                        {"shift_px": [-40, 0], "cm_per_pixel": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            code, output = self.run_cli(tello, directory, observations)
            proposal = json.loads((Path(directory) / "flight_logs/camera_calibration.json").read_text("utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(tello.calls, ["connect", "streamon", "downvision 1", "streamoff"])
        self.assertTrue(tello.reader.stopped)
        self.assertFalse(proposal["applied"])
        self.assertEqual(proposal["proposed_config"],
                         {"CAM_ROTATE_DEG": 0, "CAM_FORWARD_SIGN": 1, "CAM_LEFT_SIGN": -1})
        self.assertIn("CAM_LEFT_SIGN = -1", output)

    def test_refusal_still_stops_video_and_creates_no_proposal(self):
        tello = self.VideoOnlyTello()
        with tempfile.TemporaryDirectory() as directory:
            code, _ = self.run_cli(tello, directory, CalibrationError("mauvaise image"))
            self.assertFalse((Path(directory) / "flight_logs/camera_calibration.json").exists())
        self.assertEqual(code, 1)
        self.assertEqual(tello.calls[-1], "streamoff")
        self.assertTrue(tello.reader.stopped)

    def test_cli_source_has_no_flight_or_end_calls(self):
        source = ast.parse(Path(cli.__file__).read_text("utf-8"))
        prohibited = {"takeoff", "land", "send_rc_control", "emergency", "end",
                      "go_xyz_speed", "move_forward", "move_left", "rotate_clockwise"}
        for node in ast.walk(source):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, prohibited)


if __name__ == "__main__":
    unittest.main()
