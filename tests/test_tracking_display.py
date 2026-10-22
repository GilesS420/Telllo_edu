import math
import unittest
from unittest.mock import patch

from gui.tracking import TrajectoryHistory, display_tracking
from gui.map3d import Map3D


class TrackingDisplayTests(unittest.TestCase):
    def snapshot(self, **overrides):
        return dict(measured=True, position_valid=True, measurement_time=10, **overrides)

    def test_stale_coordinates_are_not_displayed_as_a_current_fix(self):
        mode, note = display_tracking(self.snapshot(), True, 15)
        self.assertEqual(mode, "lost")
        self.assertIn("laatste schatting", note)

    def test_relative_motion_is_visible_without_claiming_the_origin_is_known(self):
        snapshot = self.snapshot()
        snapshot["position_valid"] = False
        mode, note = display_tracking(snapshot, True, 10.1)
        self.assertEqual(mode, "local")
        self.assertIn("startpositie onbekend", note)

    def test_unmeasured_pose_is_unknown_even_if_timestamp_is_recent(self):
        snapshot = self.snapshot()
        snapshot["measured"] = False
        self.assertEqual(display_tracking(snapshot, True, 10.1)[0], "lost")

    def test_fresh_global_measurement_and_ground_are_distinguished(self):
        self.assertEqual(display_tracking(self.snapshot(), True, 10.1)[0], "global")
        self.assertEqual(display_tracking(self.snapshot(), False, 15)[0], "ground")

    def test_3d_view_marks_lost_position_and_draws_resumed_relative_motion(self):
        with patch("gui.map3d.FigureCanvasTkAgg"):
            view = Map3D(None)
        scene = dict(pose=(0, 0, 80, 0), airborne=True, position_valid=False,
                     position_mode="lost", position_note="Positie onbekend",
                     waypoints=[], mission=[], reached=0, trail=[], local_trail=[],
                     terrain=[], terrain_version=0, helipad=None, footprint=None)
        view.update(scene)
        self.assertEqual(view.nose.get_alpha(), 0.2)
        self.assertEqual(view.position_note.get_text(), "Positie onbekend")
        self.assertEqual(len(view.shadow.get_data_3d()[0]), 0)
        scene.update(pose=(30, 0, 80, 0), position_mode="local",
                     position_note="Relatieve beweging gemeten", local_trail=[(0, 0, 80), (30, 0, 80)])
        view.update(scene)
        self.assertEqual(view.nose.get_alpha(), 1.0)
        self.assertEqual(list(view.local_trail.get_data_3d()[0]), [0, 30])

    def test_tracking_loss_and_recovery_keep_previous_path_with_a_gap(self):
        history = TrajectoryHistory()
        for mode, point in (("global", (0, 0, 80)), ("global", (10, 0, 80)),
                            ("lost", (10, 0, 80)), ("local", (25, 0, 80)),
                            ("local", (30, 0, 80)), ("lost", (30, 0, 80)),
                            ("global", (35, 0, 80))):
            history.update(1, mode, point)
        self.assertEqual(history.global_points[:2], [(0, 0, 80), (10, 0, 80)])
        self.assertTrue(all(math.isnan(v) for v in history.global_points[2]))
        self.assertEqual(history.global_points[-1], (35, 0, 80))
        self.assertEqual(history.local_points, [(25, 0, 80), (30, 0, 80)])
        history.update(1, "ground", (35, 0, 0))
        self.assertEqual(len(history.global_points), 4)  # landing keeps the flight

    def test_only_a_new_coordinate_frame_or_explicit_clear_removes_history(self):
        history = TrajectoryHistory()
        points = history.global_points
        history.update(1, "global", (10, 0, 80))
        history.update(2, "global", (0, 0, 80))
        self.assertEqual(points, [(0, 0, 80)])
        history.clear()
        self.assertEqual(points, [])  # GUI and history still share the same list

    def test_plot_limits_remain_finite_with_path_gaps(self):
        with patch("gui.map3d.FigureCanvasTkAgg"):
            view = Map3D(None)
        view._limits([(0, 0, 80), (math.nan, math.nan, math.nan), (500, 0, 80)])
        self.assertTrue(all(math.isfinite(v) for v in view.ax.get_xlim()))
        self.assertGreater(view.ax.get_xlim()[1], 500)


if __name__ == "__main__":
    unittest.main()
