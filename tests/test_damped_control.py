import math
import threading
import unittest
from unittest.mock import Mock, patch

from drone import config as cfg
from drone.navigator import MotionEstimate, PathFollower, Pose


class DampedControlTests(unittest.TestCase):
    def test_takeoff_correction_stops_early_if_the_camera_direction_is_wrong(self):
        pose = Pose(10, 0, 80)
        pose.measured = True
        follower = PathFollower(Mock(), pose, threading.Event())
        clock = [0.0]

        def wrong_direction(vx, vy, vz, **kwargs):
            x, y, z, _ = pose.get()
            pose.set(x - vx / cfg.RC_HZ, y - vy / cfg.RC_HZ, z)

        follower.send_velocity = Mock(side_effect=wrong_direction)
        with patch("drone.navigator.time.time", side_effect=lambda: clock[0]), \
                patch("drone.navigator.time.sleep", side_effect=lambda dt: clock.__setitem__(0, clock[0] + dt)):
            arrived = follower.hold_position((0, 0, 80), max_speed=12, gain=0.8, damping=0.4,
                                             tolerance_cm=3, timeout_s=3, max_vertical_speed=0,
                                             max_error_growth_cm=4)
        self.assertFalse(arrived)
        self.assertLess(pose.get()[0], 16)
        self.assertEqual(follower.last_hold_result["reason"], "error_not_decreasing_check_camera_calibration")

    def test_repeated_video_sample_does_not_erase_estimated_velocity(self):
        estimate = MotionEstimate()
        estimate.observe((0, 0, 0), 0)
        velocity = estimate.observe((5, 0, 0), 0.1)
        self.assertGreater(velocity[0], 0)
        self.assertEqual(estimate.observe((5, 0, 0), 0.1), velocity)

    def test_fast_crossing_of_target_is_not_reported_as_a_settled_return(self):
        pose = Pose(0, 0, 80)
        pose.measured = True
        follower = PathFollower(Mock(), pose, threading.Event())
        follower.send_velocity = Mock()
        clock = [0.0]
        with patch("drone.navigator.time.time", side_effect=lambda: clock[0]), \
                patch("drone.navigator.time.sleep", side_effect=lambda dt: clock.__setitem__(0, clock[0] + dt)), \
                patch.object(MotionEstimate, "observe", return_value=(20, 0, 0)):
            confirmed = follower.hold_position((0, 0, 80), tolerance_cm=3,
                                               damping=0.4, timeout_s=0.5, settle_s=0.4)
        self.assertFalse(confirmed)
        self.assertLess(follower.send_velocity.call_args.args[0], 0)  # brake residual motion

    def test_xy_correction_does_not_disturb_height_already_within_tolerance(self):
        pose = Pose(20, 0, 82)
        pose.measured = True
        follower = PathFollower(Mock(), pose, threading.Event())
        follower.send_velocity = Mock(side_effect=lambda *args, **kw: pose.set(0, 0, 82))
        with patch("drone.navigator.time.sleep"):
            self.assertTrue(follower.hold_position((0, 0, 80), tolerance_cm=3, damping=0.8))
        self.assertEqual(follower.send_velocity.call_args.args[2], 0)

    def check_inertial_return(self, response_delay=0.0, inertia=0.35):
        pose = Pose(35, 9, 90)
        pose.measured = True
        tello = Mock()
        follower = PathFollower(tello, pose, threading.Event())
        clock = [0.0]
        actual = [35.0, 9.0, 90.0]
        velocity = [0.0, 0.0, 0.0]
        command = [0.0, 0.0, 0.0]
        command_history = [(-1.0, tuple(command))]
        history = [(-1.0, tuple(actual))]
        sent_vertical = []

        def sticks(lr, fb, ud, yaw):
            command[:] = [fb * 0.85, -lr * 0.85, ud * 0.85]
            command_history.append((clock[0], tuple(command)))
            sent_vertical.append(ud)

        def sleep(seconds):
            end = clock[0] + seconds
            while clock[0] < end - 1e-9:
                dt = min(0.01, end - clock[0])
                applied = next(value for when, value in reversed(command_history)
                               if when <= clock[0] - response_delay)
                for i in range(3):
                    velocity[i] += (applied[i] - velocity[i]) * dt / inertia
                    actual[i] += velocity[i] * dt
                clock[0] += dt
                history.append((clock[0], tuple(actual)))
            observed = next(position for when, position in reversed(history) if when <= clock[0] - 0.2)
            pose.set(*observed)
            pose.measurement_time = clock[0]

        tello.send_rc_control.side_effect = sticks
        with patch("drone.navigator.time.time", side_effect=lambda: clock[0]), \
                patch("drone.navigator.time.sleep", side_effect=sleep):
            confirmed = follower.hold_position(
                (0, 0, 80), max_speed=cfg.FLIP_RETURN_RC_SPEED, gain=cfg.FLIP_RETURN_GAIN,
                damping=cfg.FLIP_RETURN_DAMPING, max_vertical_speed=cfg.FLIP_RETURN_Z_SPEED,
                tolerance_cm=cfg.FLIP_RETURN_TOL_CM, timeout_s=cfg.FLIP_RETURN_TIMEOUT_S,
                settle_s=cfg.FLIP_RETURN_SETTLE_S)
        self.assertTrue(confirmed, follower.last_hold_result)
        self.assertLess(math.hypot(*actual[:2]), 6)
        self.assertLess(abs(actual[2] - 80), 6)
        self.assertLessEqual(max(abs(v) for v in sent_vertical), cfg.FLIP_RETURN_Z_SPEED)

    def test_inertia_and_delayed_video_do_not_prevent_measured_return(self):
        self.check_inertial_return()

    def test_return_settles_with_delayed_rc_response_and_video(self):
        # The flight log showed repeated overshoots despite a recovered scene.
        # Exercise input delay as well as inertia and delayed measurements.
        self.check_inertial_return(response_delay=0.35, inertia=0.4)


if __name__ == "__main__":
    unittest.main()
