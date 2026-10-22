"""Flip recovery regression checks without a drone or background threads."""

import threading
import unittest
from unittest.mock import Mock, patch

from drone import config as cfg
from drone.app import DroneApp, FlightCommandQueue
from drone.navigator import FlightInterrupted, MissionAborted, PathFollower, Pose


class FlipReturnTests(unittest.TestCase):
    def setUp(self):
        self.app = DroneApp.__new__(DroneApp)
        self.app.pose = Pose()
        self.app.pose.set(10, 20, 80, 0)
        self.app.pose.measured = True
        self.app.odometry = Mock()
        self.app.abort = threading.Event()
        self.app._gen = 0
        self.app.airborne = True
        self.app.tello = Mock()
        self.app.tello.get_battery.return_value = 90
        self.app.tello.send_command_with_return.return_value = "ok"
        self.app.follower = Mock()
        self.app.link = Mock()
        self.app.follower.command.side_effect = lambda fn, *args: fn(*args)
        self.app.follower._check_abort.side_effect = self.check_abort
        self.sleep = patch("drone.app.time.sleep").start()
        self.wait = patch.object(self.app.abort, "wait", return_value=False).start()
        self.addCleanup(patch.stopall)

    def check_abort(self):
        if self.app.abort.is_set():
            raise MissionAborted()

    def run_flip(self, **kwargs):
        self.app.handle_command({"type": "flip", **kwargs})

    def assert_target(self, target):
        self.app.follower.move_to.assert_not_called()
        self.app.follower.hold_position.assert_called_once_with(
            target, max_speed=cfg.FLIP_RETURN_RC_SPEED, gain=cfg.FLIP_RETURN_GAIN,
            tolerance_cm=cfg.FLIP_RETURN_TOL_CM, timeout_s=cfg.FLIP_RETURN_TIMEOUT_S,
            settle_s=cfg.FLIP_RETURN_SETTLE_S, damping=cfg.FLIP_RETURN_DAMPING,
            max_vertical_speed=cfg.FLIP_RETURN_Z_SPEED)

    def test_gui_xyz_returns_after_drift(self):
        def flip(*args, **kwargs):
            self.app.pose.set(55, 35, 120)
            return "ok"
        self.app.tello.send_command_with_return.side_effect = flip
        self.run_flip(pose=(10, 20, 80), spot=(10, 20))
        self.assert_target((10, 20, 80))

    def test_flip_is_sent_without_preparation_or_delay(self):
        calls_before_flip = []
        def on_flip(*args, **kwargs):
            calls_before_flip.extend(self.app.follower.mock_calls)
            return "ok"
        self.app.tello.send_command_with_return.side_effect = on_flip
        self.run_flip(pose=(10, 20, 80))
        self.app.tello.send_command_with_return.assert_called_once_with(
            "flip f", timeout=cfg.FLIP_ACK_TIMEOUT_S)
        self.app.tello.move_up.assert_not_called()
        self.sleep.assert_not_called()
        self.wait.assert_not_called()
        self.assertEqual([call[0] for call in calls_before_flip],
                         ["_check_abort", "settle_before_flip", "_check_abort"])
        self.assert_target((10, 20, 80))

    def test_altitude_only_change_still_returns(self):
        def flip(*args, **kwargs):
            self.app.pose.set(z=120)
            return "ok"
        self.app.tello.send_command_with_return.side_effect = flip
        self.run_flip(pose=(10, 20, 80))
        self.assert_target((10, 20, 80))

    def test_visual_return_uses_fast_continuous_correction(self):
        self.app.pose.measured = True
        self.run_flip(pose=(10, 20, 80))
        self.app.follower.move_to.assert_not_called()
        self.app.follower.hold_position.assert_called_once_with(
            (10, 20, 80), max_speed=cfg.FLIP_RETURN_RC_SPEED,
            gain=cfg.FLIP_RETURN_GAIN, tolerance_cm=cfg.FLIP_RETURN_TOL_CM,
            timeout_s=cfg.FLIP_RETURN_TIMEOUT_S, settle_s=cfg.FLIP_RETURN_SETTLE_S,
            damping=cfg.FLIP_RETURN_DAMPING, max_vertical_speed=cfg.FLIP_RETURN_Z_SPEED)

    def test_priority_flip_interrupts_recovery_without_being_swallowed(self):
        self.app.follower.hold_position.side_effect = FlightInterrupted()
        with self.assertRaises(FlightInterrupted):
            self.run_flip(pose=(10, 20, 80))
        self.assertFalse(self.app.follower.interruptible)
        self.app.link.send.assert_not_called()

    def test_no_gui_pose_uses_current_xyz(self):
        self.run_flip()
        self.assert_target((10, 20, 80))

    def test_sdk_error_still_attempts_return(self):
        self.app.tello.send_command_with_return.side_effect = RuntimeError("flip rejected")
        self.run_flip(pose=(10, 20, 80))
        self.assert_target((10, 20, 80))
        self.assertEqual(self.app.link.send.call_args.args[0]["ref"], "flip")

    def test_spot_without_pose_is_used(self):
        self.run_flip(spot=(5, 6))
        self.assert_target((5, 6, 80))

    def test_abort_during_settling_prevents_return(self):
        self.app.pose.measured = False
        self.wait.return_value = True
        with self.assertRaises(MissionAborted):
            self.run_flip(pose=(10, 20, 80))
        self.app.follower.move_to.assert_not_called()

    def test_abort_during_return_propagates(self):
        self.app.follower.hold_position.side_effect = MissionAborted()
        with self.assertRaises(MissionAborted):
            self.run_flip(pose=(10, 20, 80))
        self.app.follower.move_to.assert_not_called()

    def test_unknown_position_never_commands_a_fake_return(self):
        self.app.pose.position_valid = False
        with patch("drone.app.time.monotonic", side_effect=[0, 0, 0, 10]):
            self.run_flip(pose=(10, 20, 80))
        self.app.follower.move_to.assert_not_called()
        self.app.follower.hold_position.assert_not_called()
        self.assertEqual(self.app.link.send.call_args.args[0]["ref"], "flip_return")

    def test_ground_command_uses_pose_after_takeoff_reset(self):
        self.app.airborne = False
        self.app.takeoff = Mock(side_effect=lambda: self.app.pose.set(0, 0, 80))
        self.run_flip(pose=(10, 20, 0), spot=(10, 20))
        self.assert_target((0, 0, 80))


class PriorityFlipTests(unittest.TestCase):
    def test_queue_prioritizes_land_then_flips_and_preserves_fifo(self):
        commands = FlightCommandQueue()
        for kind, number in [("mission", 1), ("flip", 2), ("takeoff", 3),
                             ("flip", 4), ("land", 5)]:
            commands.put({"type": kind, "number": number})
        self.assertTrue(commands.flip_pending.is_set())
        self.assertEqual([commands.get()["number"] for _ in range(3)], [5, 2, 4])
        self.assertFalse(commands.flip_pending.is_set())
        self.assertEqual([commands.get()["number"] for _ in range(2)], [1, 3])

    def test_abort_takes_precedence_over_flip_interruption(self):
        abort = threading.Event()
        follower = PathFollower(Mock(), Pose(), abort)
        follower.flip_pending.set()
        follower._check_abort()  # not interruptible during takeoff/landing
        follower.interruptible = True
        with self.assertRaises(FlightInterrupted):
            follower._check_abort()
        abort.set()
        with self.assertRaises(MissionAborted):
            follower._check_abort()

    def test_land_submission_removes_pending_flips(self):
        app = DroneApp.__new__(DroneApp)
        app.airborne = True
        app.commands = FlightCommandQueue()
        app.abort = threading.Event()
        app._gen = 0
        app.pose = Pose(z=80)
        app.tello = Mock()
        app.submit({"type": "flip"})
        app.submit({"type": "land"})
        self.assertFalse(app.commands.flip_pending.is_set())
        self.assertTrue(app.abort.is_set())
        self.assertEqual(app.commands.get()["type"], "land")
        self.assertTrue(app.commands.empty())

    def test_worker_runs_priority_flip_after_interrupted_navigation(self):
        app = DroneApp.__new__(DroneApp)
        app.commands = FlightCommandQueue()
        app.commands.put({"type": "mission"})
        app.running = True
        app.follower = Mock()
        app.link = Mock()
        app.safe_land = Mock()
        calls = []

        def handle(cmd):
            calls.append(cmd["type"])
            if cmd["type"] == "mission":
                app.commands.put({"type": "flip"})
                raise FlightInterrupted()
            app.running = False

        app.handle_command = handle
        app.mission_loop()
        self.assertEqual(calls, ["mission", "flip"])
        app.safe_land.assert_not_called()
        app.follower.rc.assert_called_once_with(0, 0, 0, 0)

    def test_fast_hold_limits_velocity_and_stops_on_abort(self):
        abort = threading.Event()
        pose = Pose()
        pose.set(0, 0, 80, 0)
        pose.measured = True
        tello = Mock()
        follower = PathFollower(tello, pose, abort)
        follower.send_velocity = Mock(side_effect=lambda *args, **kw: abort.set())
        with patch("drone.navigator.time.sleep"):
            with self.assertRaises(MissionAborted):
                follower.hold_position((100, 0, 80), max_speed=40, gain=1.8)
        self.assertEqual(follower.send_velocity.call_args.args, (40, 0, 0))
        tello.send_rc_control.assert_called_with(0, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
