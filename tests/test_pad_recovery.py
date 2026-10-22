import math
import unittest

from drone.pad_recovery import PadRecovery


def snapshot(t=10, x=0, y=0, **flags):
    result = dict(pose=(x, y, 80, 0), measured=True, measurement_time=t,
                  position_valid=True, local_reference_valid=False)
    result.update(flags)
    return result


class PadRecoveryTests(unittest.TestCase):
    def test_return_towards_last_observed_h_is_capped_and_never_descends(self):
        recovery = PadRecovery()
        recovery.observe(snapshot(), (20, 10), 10, 10)
        velocity = recovery.velocity(snapshot(10.2, x=2), 10.2)
        self.assertGreater(velocity[0], 0)
        self.assertGreater(velocity[1], 0)
        self.assertLessEqual(math.hypot(*velocity[:2]), 6.00001)
        self.assertEqual(velocity[2], 0)
        self.assertEqual(recovery.target, (20, 10))

    def test_local_reference_can_recover_without_inventing_a_global_fix(self):
        recovery = PadRecovery()
        local = dict(position_valid=False, local_reference_valid=True)
        recovery.observe(snapshot(**local), (10, 0), 10, 10)
        self.assertGreater(recovery.velocity(snapshot(10.2, **local), 10.2)[0], 0)

    def test_unknown_stale_or_changed_reference_immediately_stops(self):
        for current in (snapshot(10.2, measured=False), snapshot(9),
                        snapshot(10.2, position_valid=False),
                        snapshot(10.2, position_valid=False, local_reference_valid=True)):
            with self.subTest(current=current):
                recovery = PadRecovery()
                recovery.observe(snapshot(), (10, 0), 10, 10)
                self.assertEqual(recovery.velocity(current, 10.2), (0, 0, 0))
                self.assertIsNone(recovery.target)

    def test_divergence_distance_and_timeout_stop_recovery(self):
        for when, x, target in ((10.3, -5, 10), (10.3, 0, 50), (13.1, 0, 10)):
            with self.subTest(when=when, x=x, target=target):
                recovery = PadRecovery()
                recovery.observe(snapshot(), (target, 0), 10, 10)
                recovery.velocity(snapshot(10.1), 10.1)
                self.assertEqual(recovery.velocity(snapshot(when, x=x), when), (0, 0, 0))

    def test_old_h_cannot_seed_a_new_recovery_or_refresh_its_timestamp(self):
        recovery = PadRecovery()
        recovery.observe(snapshot(), (10, 0), 9, 10)
        self.assertIsNone(recovery.target)
        recovery.observe(snapshot(), (10, 0), 10, 10)
        recovery.observe(snapshot(10.2, x=50), (10, 0), 10, 10.2)
        self.assertEqual(recovery.target, (10, 0))
