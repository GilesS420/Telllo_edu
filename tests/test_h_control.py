import unittest
from unittest.mock import Mock, patch

from drone import config as cfg
from drone.app import DroneApp
from drone.navigator import Pose


class HControlTests(unittest.TestCase):
    def run_control(self, observations, timeout=1):
        app = DroneApp.__new__(DroneApp)
        app.pose = Pose(0, 0, 80)
        app.pose.measured = True
        app.pose.measurement_time = 10
        app.follower = Mock()
        app._pad_near = Mock(return_value=(10, 10, 0, 0.3))
        app._live_pad = (10, 10, 0, 0.3, 80)
        clock = [10.0]
        commands = []

        def sleep(dt):
            clock[0] += dt
            app.pose.measurement_time = clock[0]
            observations(app, clock[0])

        app.follower.send_velocity.side_effect = lambda vx, vy, vz, **kw: commands.append(
            (clock[0], vx, vy, vz))
        with patch("drone.app.time.time", side_effect=lambda: clock[0]), \
                patch("drone.app.time.sleep", side_effect=sleep), \
                patch.object(cfg, "HELIPAD_TIMEOUT_S", timeout):
            result = app.land_on_helipad((10, 0))
        return result, commands, clock[0]

    def test_one_reused_frame_does_not_authorize_descent(self):
        def frames(app, now):
            # The same confirmed frame is reused until it goes stale.
            app._live_pad = (10, 1, 0, 0.3, 80)

        result, commands, _ = self.run_control(frames, timeout=0.5)
        self.assertFalse(result)
        self.assertTrue(all(vz == 0 for _, _, _, vz in commands))

    def test_dropout_recovers_towards_seen_pad_without_descending(self):
        def frames(app, now):
            app._live_pad = (now, 10, 0, 0.3, 80) if now < 10.4 else None

        result, commands, _ = self.run_control(frames)
        self.assertFalse(result)
        recovery = [cmd for cmd in commands if cmd[0] >= 10.4]
        self.assertTrue(recovery)
        self.assertTrue(all(0 < vx <= 6 and vy == vz == 0 for _, vx, vy, vz in recovery))

    def test_dropout_with_stale_position_only_hovers(self):
        def frames(app, now):
            app._live_pad = (now, 10, 0, 0.3, 80) if now < 10.4 else None
            if now >= 10.4:
                app.pose.measurement_time = 9

        result, commands, _ = self.run_control(frames)
        self.assertFalse(result)
        self.assertFalse([cmd for cmd in commands if cmd[0] >= 10.4])

    def test_correction_away_from_pad_stops_before_full_timeout(self):
        def frames(app, now):
            app._live_pad = (now, 20 if now >= 10.6 else 10, 0, 0.3, 80)

        result, _, stopped = self.run_control(frames, timeout=5)
        self.assertFalse(result)
        self.assertLess(stopped, 11)
