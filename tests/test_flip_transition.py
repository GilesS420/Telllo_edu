import threading
import unittest
from unittest.mock import Mock, patch

from drone import config as cfg
from drone.navigator import MissionAborted, PathFollower, Pose


class FlipTransitionTests(unittest.TestCase):
    def test_stable_hover_does_not_add_a_fixed_delay(self):
        follower = PathFollower(Mock(), Pose(), threading.Event())
        follower._last_rc = 10
        with patch("drone.navigator.time.time", return_value=20), \
                patch("drone.navigator.time.sleep") as sleep:
            self.assertEqual(follower.settle_before_flip(), 0)
        sleep.assert_not_called()
        follower.tello.send_rc_control.assert_not_called()

    def test_only_remaining_rc_transition_is_waited(self):
        follower = PathFollower(Mock(), Pose(), threading.Event())
        follower._last_rc = 10
        clock = [10 + cfg.RC_SETTLE_S * 0.3]
        with patch("drone.navigator.time.time", side_effect=lambda: clock[0]), \
                patch("drone.navigator.time.sleep", side_effect=lambda dt: clock.__setitem__(0, clock[0] + dt)):
            elapsed = follower.settle_before_flip()
        self.assertAlmostEqual(elapsed, cfg.RC_SETTLE_S * 0.7, places=5)
        follower.tello.send_rc_control.assert_called_once_with(0, 0, 0, 0)
        self.assertEqual(follower._last_rc, 10)

    def test_land_interrupts_transition_without_waiting_for_the_delay(self):
        abort = threading.Event()
        abort.set()
        follower = PathFollower(Mock(), Pose(), abort)
        follower._last_rc = 10
        with patch("drone.navigator.time.time", return_value=10.1), \
                patch("drone.navigator.time.sleep") as sleep:
            with self.assertRaises(MissionAborted):
                follower.settle_before_flip()
        sleep.assert_not_called()
