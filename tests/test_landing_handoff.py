import threading
import unittest
from unittest.mock import Mock, patch

from drone import config as cfg
from drone.app import DroneApp
from drone.navigator import PathFollower, Pose


class LandingHandoffTests(unittest.TestCase):
    def attempt(self, height, offset, size, sensor_height=None):
        app = DroneApp.__new__(DroneApp)
        app.pose = Pose(0, 0, height)
        app.pose.measured = True
        app.follower = Mock()
        if sensor_height is not None:
            app.read_height = Mock(return_value=sensor_height)
        self.sent = []
        app.follower.send_velocity.side_effect = lambda vx, vy, vz, **kw: self.sent.append((vx, vy, vz))
        clock = [10.0]
        app._pad_near = Mock(return_value=(10.0, 0, 0, size))
        app._live_pad = (10.0, offset, 0, size, height)

        def sleep(seconds):
            clock[0] += seconds
            app._live_pad = (clock[0], offset, 0, size, height)

        with patch("drone.app.time.time", side_effect=lambda: clock[0]), \
                patch("drone.app.time.sleep", side_effect=sleep), \
                patch.object(cfg, "HELIPAD_TIMEOUT_S", 1):
            return app.land_on_helipad((0, 0))

    def test_large_marker_does_not_trigger_final_land_high_above_ground(self):
        self.assertFalse(self.attempt(90, 1, 0.9))

    def test_low_centered_pad_allows_final_land_after_stable_observations(self):
        self.assertTrue(self.attempt(cfg.HELIPAD_FINAL_CM, 1, 0.5))

    def test_low_but_off_center_does_not_allow_final_land(self):
        self.assertFalse(self.attempt(cfg.HELIPAD_FINAL_CM, 5, 0.9))

    def test_latest_height_slows_descent_before_delayed_video_reaches_handoff(self):
        self.assertFalse(self.attempt(80, 1, 0.5, sensor_height=cfg.HELIPAD_FINAL_CM + 3))
        descents = [vz for _, _, vz in self.sent if vz < 0]
        self.assertTrue(descents)
        self.assertTrue(all(abs(vz) <= 3 for vz in descents))

    def test_hover_packets_do_not_restart_the_sdk_settling_delay(self):
        follower = PathFollower(Mock(), Pose(), threading.Event())
        with patch("drone.navigator.time.time", return_value=10):
            follower.rc(0, 15, 0, 0)
        with patch("drone.navigator.time.time", return_value=11):
            follower.rc(0, 0, 0, 0)
            with patch("drone.navigator.time.sleep") as sleep:
                follower.command(follower.tello.land)
        sleep.assert_not_called()
        follower.tello.land.assert_called_once()
