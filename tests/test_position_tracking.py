import collections
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from drone import config as cfg
from drone.app import CommandRejected, DroneApp
from drone.navigator import OdometryLost, PathFollower, Pose
from drone.odometry import Odometry


class VisualTrackingTests(unittest.TestCase):
    def setUp(self):
        self.vo = Odometry.__new__(Odometry)
        self.vo.pose = Pose(z=80)
        self.vo._window = None
        self.vo._control_lock = threading.RLock()
        self.vo._flip_active = False
        self.vo._flip_ref = None
        self.vo._reacquire = False
        self.vo._resume_at = 0
        self.vo._world = (0, 0)
        self.vo._last_good = 10
        self.vo._prev_frame = None
        self.vo.tello = Mock()
        self.vo.tello.get_yaw.return_value = 0
        self.vo.tello.get_distance_tof.return_value = 80
        self.vo.tello.get_current_state.return_value = {"pitch": 0, "roll": 0}
        self.vo.frame_reader = Mock()
        self.vo.is_airborne = lambda: True
        self.vo.downcam_on = lambda: True
        self.vo._gray = lambda frame: frame
        rng = np.random.default_rng(19)
        self.image = cv2.GaussianBlur(rng.uniform(0, 255, (240, 320)).astype(np.float32), (0, 0), 1)
        self.ref = dict(img=self.image.copy(), yaw=0, height=80, world=(0, 0))
        self.vo._key = self.vo._prev = self.ref
        self.vo._good_frames = collections.deque([(10, self.ref, 0.8)], maxlen=6)

    def test_vertical_climb_scale_does_not_invent_horizontal_drift(self):
        # Higher camera -> smaller floor features about the optical centre.
        transform = cv2.getRotationMatrix2D((159.5, 119.5), 0, 80 / 100)
        climbed = cv2.warpAffine(self.image, transform, (320, 240))
        shift, quality = self.vo._compare(self.ref, climbed, height=100)
        self.assertGreater(quality, cfg.VO_MIN_RESPONSE)
        self.assertLess(np.linalg.norm(shift), 0.6)

    def test_unrelated_frames_are_rejected(self):
        noise = np.random.default_rng(88).uniform(0, 255, (240, 320)).astype(np.float32)
        _, quality = self.vo._compare(self.ref, noise, height=80)
        self.assertEqual(quality, 0)

    def test_image_matching_does_not_mutate_reference(self):
        before = self.ref["img"].copy()
        for _ in range(5):
            self.vo._compare(self.ref, self.image, height=80)
        np.testing.assert_array_equal(before, self.ref["img"])

    def test_flip_rotation_never_updates_xy_or_height(self):
        self.vo.pose.measured = True
        self.vo.begin_flip()
        self.vo.tello.get_distance_tof.return_value = 500
        self.vo.frame_reader.frame = np.zeros_like(self.image)
        self.vo._step(11, 0.05)
        self.assertEqual(self.vo.pose.get()[:3], (0, 0, 80))
        self.assertFalse(self.vo.pose.position_valid)

    def test_preflip_reference_recovers_actual_displacement(self):
        self.vo.pose.measured = True
        self.vo.begin_flip()
        self.vo.end_flip()
        self.vo._resume_at = 10
        self.vo.frame_reader.frame = cv2.warpAffine(
            self.image, np.float32([[1, 0, 0], [0, 1, 12]]), (320, 240))
        self.vo._step(11, 0.05)
        self.assertFalse(self.vo.pose.measured)  # one image cannot authorize a return
        self.vo.frame_reader.frame = self.vo.frame_reader.frame.copy()
        self.vo._step(11.05, 0.05)
        self.assertTrue(self.vo.pose.position_valid)
        self.assertTrue(self.vo.pose.measured)
        self.assertGreater(np.linalg.norm(self.vo.pose.get()[:2]), 1)
        self.assertFalse(self.vo._reacquire)

    def test_no_matching_floor_keeps_global_position_unknown(self):
        self.vo.pose.measured = True
        self.vo.begin_flip()
        self.vo.end_flip()
        self.vo._resume_at = 10
        self.vo.frame_reader.frame = np.zeros_like(self.image)
        self.vo._step(11, 0.05)
        self.assertFalse(self.vo.pose.position_valid)
        self.assertFalse(self.vo.pose.measured)
        self.assertEqual(self.vo.pose.get()[:2], (0, 0))

    def test_local_click_reference_works_after_the_flight_origin_was_lost(self):
        self.vo.pose.set(10, -2, 80)
        self.vo.pose.measured = True
        self.vo.pose.position_valid = False
        with patch("drone.odometry.time.time", return_value=10.1):
            captured = self.vo.capture_flip_reference()
        self.assertIsNotNone(captured["reference"])
        self.assertFalse(captured["snapshot"]["tracking_valid"])
        # The queue may wait while the previous flight command completes.
        self.vo.pose.set(50, 20, 80)
        self.vo._world = (15, 8)
        self.vo._prev = dict(self.ref, world=(15, 8))
        self.vo.begin_flip(captured["reference"])
        self.vo.end_flip()
        self.vo._resume_at = 10
        self.vo.frame_reader.frame = cv2.warpAffine(
            self.image, np.float32([[1, 0, 0], [0, 1, 12]]), (320, 240))
        self.vo._step(11, 0.05)
        self.vo.frame_reader.frame = self.vo.frame_reader.frame.copy()
        self.vo._step(11.05, 0.05)
        self.assertTrue(self.vo.pose.measured)
        self.assertTrue(self.vo.pose.local_reference_valid)
        self.assertFalse(self.vo.pose.position_valid)  # must not invent a recovered global origin
        shift, _ = self.vo._compare(self.ref, self.vo.frame_reader.frame, 80)
        expected_delta = self.vo._position(self.ref, shift, 320, 80)
        np.testing.assert_allclose(self.vo.pose.get()[:2],
                                   (10 + expected_delta[0], -2 + expected_delta[1]), atol=0.1)

    def test_stale_floor_image_cannot_be_used_as_a_click_reference(self):
        self.vo.pose.measured = True
        with patch("drone.odometry.time.time", return_value=11):
            self.assertIsNone(self.vo.capture_flip_reference()["reference"])

    def test_click_image_stays_the_reference_during_the_entire_return(self):
        self.vo.pose.measured = True
        self.vo.pose.position_valid = False
        with patch("drone.odometry.time.time", return_value=10.1):
            capture = self.vo.capture_flip_reference()["reference"]
        self.vo.begin_flip(capture)
        self.vo.end_flip()
        self.vo._resume_at = 10
        for t in (11, 11.05):
            self.vo.frame_reader.frame = cv2.warpAffine(
                self.image, np.float32([[1, 0, 0], [0, 1, 12]]), (320, 240))
            self.vo._step(t, 0.05)
        self.assertTrue(self.vo._anchor_tracking)
        # Simulate a bad incremental estimate: the saved scene must override
        # this drift on the next frame, not integrate another delta into it.
        self.vo.pose.set(999, -999, 80)
        self.vo.frame_reader.frame = cv2.warpAffine(
            self.image, np.float32([[1, 0, 0], [0, 1, 6]]), (320, 240))
        self.vo._step(11.1, 0.05)
        expected = self.vo._position(self.ref, (0, 6), 320, 80)
        np.testing.assert_allclose(self.vo.pose.get()[:2], expected, atol=0.3)
        self.vo.finish_flip_recovery()
        self.assertFalse(self.vo._anchor_tracking)
        self.assertFalse(self.vo.pose.position_valid)

    def test_capture_ignores_an_unmatched_blurred_previous_frame(self):
        self.vo._prev = dict(self.ref, img=np.zeros_like(self.image))
        with patch("drone.odometry.time.time", return_value=10.1):
            captured = self.vo.capture_flip_reference()["reference"]
        np.testing.assert_array_equal(captured["frame"]["img"], self.image)
        self.assertEqual(captured["quality"], 0.8)

    def test_rotation_and_height_change_recover_a_correct_floor_offset(self):
        self.vo.pose.measured = True
        self.vo.begin_flip()
        matrix = cv2.getRotationMatrix2D((159.5, 119.5), -12, 80 / 105)
        matrix[:, 2] += (18, 7)
        frame = cv2.warpAffine(self.image, matrix, (320, 240))
        found, _ = self.vo._recover_position(frame, 105, 12)
        self.assertIsNotNone(found)
        inverse = cv2.invertAffineTransform(matrix)
        ref_centre = inverse @ np.array([159.5, 119.5, 1])
        expected = self.vo._position(self.ref, np.array([159.5, 119.5]) - ref_centre, 320, 80)
        np.testing.assert_allclose(found, expected, atol=0.8)

    def test_feature_registration_recovers_large_translation_beyond_phase_limit(self):
        self.vo.begin_flip()
        frame = cv2.warpAffine(self.image, np.float32([[1, 0, 120], [0, 1, 12]]), (320, 240))
        shift, detail = self.vo._feature_shift(self.ref, frame, 80, 0)
        self.assertIsNotNone(shift, detail)
        np.testing.assert_allclose(shift, (120, 12), atol=1)
        self.assertGreaterEqual(detail["inliers"], 12)

    def test_normal_tracking_measures_large_translation_instead_of_freezing_xy(self):
        self.vo.frame_reader.frame = cv2.warpAffine(
            self.image, np.float32([[1, 0, 120], [0, 1, 12]]), (320, 240))
        self.vo._step(10.5, 0.5)
        expected = self.vo._position(self.ref, (120, 12), 320, 80)
        np.testing.assert_allclose(self.vo.pose.get()[:2], expected, atol=0.5)
        self.assertTrue(self.vo.pose.measured)
        self.assertTrue(self.vo.pose.position_valid)

    def test_feature_fallback_does_not_accept_impossible_instantaneous_jump(self):
        self.vo.frame_reader.frame = cv2.warpAffine(
            self.image, np.float32([[1, 0, 120], [0, 1, 12]]), (320, 240))
        self.vo._step(10.02, 0.02)
        np.testing.assert_allclose(self.vo.pose.get()[:2], (0, 0))
        self.assertFalse(self.vo.pose.measured)
        self.assertFalse(self.vo.pose.position_valid)

    def test_feature_registration_handles_small_perspective_change(self):
        matrix = np.float32([[1, 0.04, 18], [0.02, 1, 8], [0.00015, -0.0001, 1]])
        frame = cv2.warpPerspective(self.image, matrix, (320, 240))
        shift, detail = self.vo._feature_shift(self.ref, frame, 80, 0)
        self.assertIsNotNone(shift, detail)
        point = cv2.perspectiveTransform(np.float32([[[159.5, 119.5]]]), np.linalg.inv(matrix))[0, 0]
        np.testing.assert_allclose(shift, np.array([159.5, 119.5]) - point, atol=1)

    def test_feature_registration_retains_overlap_after_scale_rotation_and_large_shift(self):
        # Resampling before descriptor extraction used to lose detail and clip
        # the edge of the scene after a combined climb and lateral displacement.
        image = cv2.GaussianBlur(np.random.default_rng(82).uniform(
            0, 255, (240, 320)).astype(np.float32), (0, 0), 0.7)
        ref = dict(img=image, yaw=0, height=80, world=(0, 0))
        matrix = cv2.getRotationMatrix2D((159.5, 119.5), -14, 80 / 120)
        matrix[:, 2] += (70, 10)
        frame = cv2.warpAffine(image, matrix, (320, 240))
        shift, detail = self.vo._feature_shift(ref, frame, 120, 14)
        self.assertIsNotNone(shift, detail)
        expected_point = cv2.invertAffineTransform(matrix) @ np.array([159.5, 119.5, 1])
        np.testing.assert_allclose(shift, np.array([159.5, 119.5]) - expected_point, atol=1)
        self.assertGreaterEqual(detail["inliers"], 12)

    def test_feature_registration_still_rejects_geometry_inconsistent_with_sensors(self):
        matrix = cv2.getRotationMatrix2D((159.5, 119.5), 40, 1)
        frame = cv2.warpAffine(self.image, matrix, (320, 240))
        shift, detail = self.vo._feature_shift(self.ref, frame, 80, 0)
        self.assertIsNone(shift)
        self.assertEqual(detail["reason"], "plane_disagrees_with_imu_or_height")

    def test_feature_registration_rejects_unrelated_and_blank_images(self):
        for frame in (np.zeros_like(self.image),
                      np.random.default_rng(51).uniform(0, 255, self.image.shape).astype(np.float32)):
            shift, _ = self.vo._feature_shift(self.ref, frame, 80, 0)
            self.assertIsNone(shift)

    def test_registration_debug_images_and_reason_are_saved(self):
        self.vo.pose.measured = True
        self.vo.begin_flip()
        self.vo._recovery_image = self.image.copy()
        with tempfile.TemporaryDirectory() as directory:
            detail = self.vo.save_flip_diagnostics(directory)
            for name in ("reference_image", "after_image"):
                saved = cv2.imread(str(Path(directory) / detail[name]), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(saved.shape, self.image.shape)
        self.assertTrue(detail["reference_saved"])

    def test_invalid_height_is_reported_as_sensor_problem(self):
        self.vo.pose.measured = True
        self.vo.begin_flip()
        self.vo.end_flip()
        self.vo._resume_at = 10
        self.vo.tello.get_distance_tof.return_value = 10
        self.vo.frame_reader.frame = self.image.copy()
        self.vo._step(11, 0.05)
        self.assertEqual(self.vo._recovery_diag["reason"], "invalid_height")
        self.assertFalse(self.vo.pose.local_reference_valid)

    def test_tilted_camera_is_not_used_as_horizontal_translation(self):
        self.vo.tello.get_current_state.return_value = {"pitch": 70, "roll": 0}
        self.vo._step(11, 0.05)
        self.assertFalse(self.vo.pose.measured)
        self.assertEqual(self.vo.pose.get()[:2], (0, 0))


class HelipadAndEstimateTests(unittest.TestCase):
    def setUp(self):
        self.app = DroneApp.__new__(DroneApp)
        self.app.pose = Pose(z=80)
        self.app.pose.measured = True
        self.app.known_pad = None
        self.app._live_pad = None
        self.app.pads = [(time.time(), 5, 8, 0.2)] * cfg.HELIPAD_MIN_SIGHTINGS
        self.app.takeoff = Mock()
        self.app.follower = Mock()
        self.app.safe_land = Mock()

    def test_confirmed_h_survives_history_eviction(self):
        self.assertEqual(self.app.seen_helipad(), (5, 8))
        self.app.pads = []
        self.assertEqual(self.app.seen_helipad(), (5, 8))

    def test_h_button_uses_remembered_coordinates(self):
        self.app.land_on_helipad = Mock(return_value=True)
        self.app.helipad_test()
        self.app.follower.move_to.assert_called_once_with((5, 8, 80), cfg.DEFAULT_SPEED,
                                                         require_tracking=True)
        self.app.land_on_helipad.assert_called_once_with((5, 8))
        self.assertTrue(self.app.pads)
        self.app.safe_land.assert_called_once_with(helipad=False)

    def test_visible_h_bypasses_remembered_coordinate_approach(self):
        self.app.pose.position_valid = False
        self.app.pose.local_reference_valid = True
        self.app.local_known_pad = (100, 100)  # an old estimate must not override the actual H
        self.app.local_pads = []
        self.app._live_pad = (time.time(), 2, 3, 0.3, 80)
        self.app.land_on_helipad = Mock(return_value=True)
        self.app.helipad_test()
        self.app.follower.hold_position.assert_not_called()
        self.app.follower.move_to.assert_not_called()
        self.app.land_on_helipad.assert_called_once_with((0, 0))
        self.app.safe_land.assert_called_once_with(helipad=False)

    def test_remembered_h_approach_timeout_scales_with_distance(self):
        self.app.pose.position_valid = False
        self.app.pose.local_reference_valid = True
        self.app.local_known_pad = (0, 60)
        self.app.local_pads = []
        self.app.follower.hold_position.return_value = True
        self.app.land_on_helipad = Mock(return_value=True)
        self.app.helipad_test()
        arguments = self.app.follower.hold_position.call_args.kwargs
        self.assertGreater(arguments["timeout_s"], cfg.FINE_TIMEOUT_S)
        self.assertLessEqual(arguments["timeout_s"], cfg.HELIPAD_APPROACH_MAX_S)
        self.assertGreater(arguments["damping"], 0)

    def test_h_appearing_during_approach_continues_visually_after_timeout(self):
        self.app.pose.position_valid = False
        self.app.pose.local_reference_valid = True
        self.app.local_known_pad = (0, 60)
        self.app.local_pads = []

        def approach(*args, **kwargs):
            self.app._live_pad = (time.time(), 2, 3, 0.3, 80)
            return False

        self.app.follower.hold_position.side_effect = approach
        self.app.land_on_helipad = Mock(return_value=True)
        self.app.helipad_test()
        self.app.land_on_helipad.assert_called_once()
        self.app.safe_land.assert_called_once_with(helipad=False)

    def test_h_button_does_not_land_elsewhere_when_h_missing(self):
        self.app.land_on_helipad = Mock(return_value=False)
        with self.assertRaises(CommandRejected):
            self.app.helipad_test()
        self.app.safe_land.assert_not_called()

    def test_unknown_position_does_not_navigate_to_old_h_coordinates(self):
        self.app.pose.position_valid = False
        self.app.land_on_helipad = Mock(return_value=False)
        with self.assertRaises(CommandRejected):
            self.app.helipad_test()
        self.app.follower.move_to.assert_not_called()

    def test_blind_command_completion_is_marked_as_an_estimate(self):
        pose = Pose(z=80)
        pose.begin_move((100, 0, 80), 30)
        pose.end_move()
        self.assertFalse(pose.position_valid)

    def test_h_return_stops_if_tracking_is_lost(self):
        follower = PathFollower(Mock(), Pose(z=80), threading.Event())
        with self.assertRaises(OdometryLost):
            follower.move_to((100, 0, 80), 30, require_tracking=True)
        follower.tello.go_xyz_speed.assert_not_called()

    def test_lost_live_h_stops_instead_of_claiming_success(self):
        self.app._pad_near = Mock(return_value=(time.time(), 5, 8, 0.2))
        self.app._live_pad = (time.time() - 4, 10, 0, 0.2, 80)
        self.assertFalse(self.app.land_on_helipad((0, 0)))
        self.app.follower.send_velocity.assert_not_called()

    def test_h_search_does_not_announce_or_attempt_climb_after_position_loss(self):
        self.app.pose.position_valid = False
        self.app.known_pad = (2, -2)
        self.app._pad_near = Mock(return_value=None)
        self.app._look_for_pad = Mock(return_value=None)
        with patch("builtins.print") as output:
            self.assertFalse(self.app.land_on_helipad((0, 0)))
        self.app._look_for_pad.assert_called_once_with((0, 0), 0.0)
        self.assertEqual(self.app.known_pad, (2, -2))
        self.assertFalse(any("stijgen naar" in str(call) for call in output.call_args_list))

    def test_h_search_can_expand_view_when_global_tracking_is_available(self):
        self.app.pose.position_valid = True
        self.app._pad_near = Mock(return_value=None)
        self.app._look_for_pad = Mock(return_value=None)
        self.assertFalse(self.app.land_on_helipad((0, 0)))
        self.assertEqual(self.app._look_for_pad.call_count, 2)
        self.assertEqual(self.app._look_for_pad.call_args.args, ((0, 0), cfg.HELIPAD_CLIMB_CMS))


if __name__ == "__main__":
    unittest.main()
