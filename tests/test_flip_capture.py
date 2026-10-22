import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from drone import config as cfg
from drone.app import DroneApp, FlightCommandQueue
from drone.navigator import PathFollower, Pose


class FlipCaptureTests(unittest.TestCase):
    def setUp(self):
        self.app = DroneApp.__new__(DroneApp)
        self.app.pose = Pose(12, -7, 85, 0)
        self.app.pose.measured = True
        self.app._gen = 0
        self.app.commands = FlightCommandQueue()
        self.app.abort = threading.Event()
        self.app.airborne = True
        self.app.follower = Mock()
        self.app.follower.hold_position.return_value = True
        self.app.tello = Mock()
        self.app.tello.get_battery.return_value = 90
        self.app.tello.send_command_with_return.return_value = "ok"
        self.app.odometry = Mock()
        self.app.odometry.capture_flip_reference.side_effect = lambda: {
            "snapshot": self.app.pose.snapshot(), "reference": {"test_reference": True}}
        self.app.link = Mock()

    def test_each_click_keeps_its_own_target_while_queue_waits(self):
        self.app.submit({"type": "flip", "dir": "f"})
        first = self.app.commands.get()
        self.app.pose.set(25, 10, 90)
        self.app.submit({"type": "flip", "dir": "b"})
        second = self.app.commands.get()
        self.assertEqual(first["pose"], (12, -7, 85))
        self.assertEqual(first["_flip_record"]["target_cm"], (12, -7, 85))
        self.assertEqual(second["pose"], (25, 10, 90))
        self.assertIsNot(first["_flip_record"], second["_flip_record"])

    def test_flip_interrupt_is_signalled_before_waiting_for_camera_capture(self):
        pending_at_capture = []

        def capture():
            pending_at_capture.append(self.app.commands.flip_pending.is_set())
            return {"snapshot": self.app.pose.snapshot(), "reference": {"test_reference": True}}

        self.app.odometry.capture_flip_reference.side_effect = capture
        self.app._motion_phase = "takeoff_recenter"
        self.app.submit({"type": "flip"})
        self.assertEqual(pending_at_capture, [True])
        self.assertEqual(self.app.commands.get()["_waiting_for"], "takeoff_recenter")

    def test_invalid_flip_does_not_leave_a_navigation_interrupt(self):
        self.app.submit({"type": "flip", "pose": [float("nan"), 0, 80]})
        self.assertFalse(self.app.commands.flip_pending.is_set())
        self.assertTrue(self.app.commands.empty())

    def test_sdk_refusal_does_not_move_back_to_the_old_click_position(self):
        self.app._save_flip_record = Mock()
        self.app.tello.send_command_with_return.return_value = "error battery low"
        self.app.submit({"type": "flip"})
        self.app.handle_command(self.app.commands.get())
        record = self.app.last_flip_record
        self.assertEqual(record["status"], "flip_failed")
        self.assertFalse(record["sdk_accepted"])
        self.assertEqual(record["return_status"], "not_requested_sdk_refused")
        self.assertIn("battery low", record["sdk_error"])
        self.app.follower.hold_position.assert_not_called()
        self.app.follower.move_to.assert_not_called()
        self.app.follower.rc.assert_called_once_with(0, 0, 0, 0)
        self.app.tello.send_command_with_return.assert_called_once()

    def test_flip_click_during_takeoff_is_rejected_without_storing_a_moving_target(self):
        self.app.state = "taking_off"
        self.app._motion_phase = "takeoff_sdk"
        self.app.submit({"type": "flip"})
        self.assertTrue(self.app.commands.empty())
        self.assertFalse(self.app.commands.flip_pending.is_set())
        self.app.odometry.capture_flip_reference.assert_not_called()
        self.app.tello.send_command_with_return.assert_not_called()
        self.assertEqual(self.app.link.send.call_args.args[0]["ref"], "flip")

    def test_plain_sdk_error_also_skips_return_movement(self):
        self.app._save_flip_record = Mock()
        self.app.tello.send_command_with_return.return_value = "error"
        self.app.submit({"type": "flip"})
        cmd = self.app.commands.get()
        self.app.pose.set(25, 10, 82)
        self.app.handle_command(cmd)
        self.app.follower.hold_position.assert_not_called()
        self.assertEqual(self.app.last_flip_record["status"], "flip_failed")
        self.assertEqual(self.app.last_flip_record["sdk_response"], "error")

    def test_dispatch_clock_includes_camera_capture_delay(self):
        clock = [10.0]

        def capture():
            clock[0] += 0.3
            return {"snapshot": self.app.pose.snapshot(), "reference": {"test_reference": True}}

        self.app.odometry.capture_flip_reference.side_effect = capture
        with patch("drone.app.time.monotonic", side_effect=lambda: clock[0]):
            self.app.submit({"type": "flip"})
        self.assertEqual(self.app.commands.get()["_submitted"], 10.0)

    def test_capture_and_confirmed_result_are_saved_after_sdk_send(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_flip.json"
            self.app.submit({"type": "flip", "dir": "f"})
            cmd = self.app.commands.get()
            existed_at_send = []
            self.app.tello.send_command_with_return.side_effect = (
                lambda *args, **kw: existed_at_send.append(path.exists()) or "ok")
            with patch.object(cfg, "FLIP_RECORD_PATH", str(path)):
                self.app.handle_command(cmd)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(existed_at_send, [False])
            self.assertEqual(saved["target_cm"], [12, -7, 85])
            self.assertTrue(saved["tracking_valid_at_click"])
            self.assertEqual(saved["status"], "confirmed")
            self.assertEqual(saved["xy_error_cm"], 0)

    def test_failed_precision_does_not_claim_return_even_at_target(self):
        self.app.follower.hold_position.return_value = False
        self.app._save_flip_record = Mock()
        self.app.submit({"type": "flip"})
        self.app.handle_command(self.app.commands.get())
        self.assertEqual(self.app.last_flip_record["status"], "not_confirmed")

    def test_recent_local_click_reference_allows_return_with_invalid_global_origin(self):
        self.app.pose.position_valid = False
        reference = {"test_reference": True}
        self.app.odometry.capture_flip_reference.side_effect = None
        self.app.odometry.capture_flip_reference.return_value = {
            "snapshot": self.app.pose.snapshot(), "reference": reference}
        self.app.odometry.end_flip.side_effect = lambda: setattr(self.app.pose, "local_reference_valid", True)
        self.app._save_flip_record = Mock()
        self.app.submit({"type": "flip"})
        self.app.handle_command(self.app.commands.get())
        self.app.odometry.begin_flip.assert_called_once_with(reference)
        self.app.follower.hold_position.assert_called_once_with(
            (12, -7, 85), max_speed=cfg.FLIP_RETURN_RC_SPEED, gain=cfg.FLIP_RETURN_GAIN,
            tolerance_cm=cfg.FLIP_RETURN_TOL_CM, timeout_s=cfg.FLIP_RETURN_TIMEOUT_S,
            settle_s=cfg.FLIP_RETURN_SETTLE_S, allow_local_reference=True,
            damping=cfg.FLIP_RETURN_DAMPING, max_vertical_speed=cfg.FLIP_RETURN_Z_SPEED)
        self.assertTrue(self.app.last_flip_record["visual_reference_at_click"])
        self.assertFalse(self.app.last_flip_record["tracking_valid_at_click"])
        self.assertEqual(self.app.last_flip_record["status"], "confirmed")
        self.assertEqual(self.app.last_flip_record["return_reference"], "click_image")
        self.assertFalse(self.app.pose.position_valid)

    def test_missing_local_and_global_reference_reports_saved_but_unusable_position(self):
        self.app.pose.position_valid = False
        self.app._save_flip_record = Mock()
        self.app.odometry.capture_flip_reference.side_effect = lambda: {
            "snapshot": self.app.pose.snapshot(), "reference": None}
        self.assertFalse(self.app.submit({"type": "flip"}))
        self.assertTrue(self.app.commands.empty())
        self.assertFalse(self.app.commands.flip_pending.is_set())
        self.app.tello.send_command_with_return.assert_not_called()
        self.app.follower.hold_position.assert_not_called()
        self.assertEqual(self.app.last_flip_record["target_cm"], (12, -7, 85))
        self.assertEqual(self.app.last_flip_record["status"], "rejected_before_sdk")


class PrecisionControlTests(unittest.TestCase):
    def test_local_reference_is_only_used_when_explicitly_authorized(self):
        pose = Pose(0, 0, 80)
        pose.measured = True
        pose.position_valid = False
        pose.local_reference_valid = True
        follower = PathFollower(Mock(), pose, threading.Event())
        self.assertFalse(follower.hold_position((0, 0, 80)))
        self.assertTrue(follower.hold_position((0, 0, 80), allow_local_reference=True))

    def test_brief_tracking_loss_pauses_and_resumes_return(self):
        pose = Pose(15, 0, 80)
        pose.measured = True
        follower = PathFollower(Mock(), pose, threading.Event())
        clock = [0.0]

        def velocity(*args, **kwargs):
            pose.set(0, 0, 80)
            pose.measured = False

        def sleep(seconds):
            clock[0] += seconds
            if clock[0] > 0.25:
                pose.measured = True

        follower.send_velocity = Mock(side_effect=velocity)
        with patch("drone.navigator.time.time", side_effect=lambda: clock[0]), \
                patch("drone.navigator.time.sleep", side_effect=sleep):
            self.assertTrue(follower.hold_position((0, 0, 80), tolerance_cm=3,
                                                   timeout_s=4, settle_s=0.4))
        self.assertGreaterEqual(clock[0], 0.65)
        follower.send_velocity.assert_called_once()

    def test_return_finishes_quickly_when_drone_ignores_tiny_stick_commands(self):
        pose = Pose(35, 0, 80)
        pose.measured = True
        tello = Mock()
        follower = PathFollower(tello, pose, threading.Event())
        clock = [0.0]
        commands = []

        def sticks(lr, fb, ud, yaw):
            commands.append((lr, fb, ud))
            if max(abs(lr), abs(fb), abs(ud)) < cfg.RC_MIN_UNITS:
                return  # reproduce the dead zone that stalled the old final approach
            x, y, z, _ = pose.get()
            pose.set(x + fb / cfg.RC_HZ, y - lr / cfg.RC_HZ, z + ud / cfg.RC_HZ)

        tello.send_rc_control.side_effect = sticks
        with patch("drone.navigator.time.time", side_effect=lambda: clock[0]), \
                patch("drone.navigator.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            confirmed = follower.hold_position(
                (0, 0, 80), max_speed=cfg.FLIP_RETURN_RC_SPEED, gain=cfg.FLIP_RETURN_GAIN,
                tolerance_cm=cfg.FLIP_RETURN_TOL_CM, timeout_s=cfg.FLIP_RETURN_TIMEOUT_S,
                settle_s=cfg.FLIP_RETURN_SETTLE_S)
        self.assertTrue(confirmed)
        self.assertLess(clock[0], 3)
        self.assertLess(abs(pose.get()[0]), cfg.FLIP_RETURN_TOL_CM)
        self.assertLessEqual(max(abs(fb) for _, fb, _ in commands), cfg.FLIP_RETURN_RC_SPEED)

    def test_small_height_error_is_corrected_despite_mission_deadband(self):
        pose = Pose(20, 0, 90)
        pose.measured = True
        follower = PathFollower(Mock(), pose, threading.Event())
        clock = [0.0]
        sent = []

        def velocity(vx, vy, vz, **kwargs):
            sent.append((vx, vy, vz))
            x, y, z, _ = pose.get()
            pose.set(x + vx / cfg.RC_HZ, y + vy / cfg.RC_HZ, z + vz / cfg.RC_HZ)

        follower.send_velocity = velocity
        with patch("drone.navigator.time.time", side_effect=lambda: clock[0]), \
                patch("drone.navigator.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            confirmed = follower.hold_position((0, 0, 80), max_speed=25, gain=1.2,
                                               tolerance_cm=3, timeout_s=8, settle_s=0.4)
        self.assertTrue(confirmed)
        self.assertTrue(any(vz < 0 for _, _, vz in sent))
        self.assertLess(math.hypot(*pose.get()[:2]), 3)
        self.assertLess(abs(pose.get()[2] - 80), 3)
        self.assertGreater(clock[0], 0.4)

    def test_already_at_target_still_requires_stable_observations(self):
        pose = Pose(0, 0, 80)
        pose.measured = True
        follower = PathFollower(Mock(), pose, threading.Event())
        clock = [0.0]
        with patch("drone.navigator.time.time", side_effect=lambda: clock[0]), \
                patch("drone.navigator.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            self.assertTrue(follower.hold_position((0, 0, 80), tolerance_cm=3, settle_s=0.4))
        self.assertGreaterEqual(clock[0], 0.4)


if __name__ == "__main__":
    unittest.main()
