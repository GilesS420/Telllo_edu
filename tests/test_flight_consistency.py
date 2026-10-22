import threading
import unittest
from unittest.mock import Mock

from drone.app import CommandRejected, DroneApp, FlightCommandQueue
from drone.downcam import DownCam
from drone.helipad import HelipadDetector, HelipadTracker
from drone.navigator import Pose
from drone.telemetry import Telemetry, TerrainMap
from tests import test_helipad_tracking as image_fixtures


class FlightConsistencyTests(unittest.TestCase):
    def relocalization_app(self):
        app = DroneApp.__new__(DroneApp)
        app.pose = Pose(50, 20, 80)
        app.pose.measured = True
        app.pose.position_valid = False
        app.pose.local_reference_valid = True
        app.known_pad = (10, 10)
        app._active_command = None
        app.commands = FlightCommandQueue()
        app.odometry = Mock()
        app.odometry.relocalize_from_pad.side_effect = app.pose.relocalize_xy
        return app

    def test_known_h_restores_global_frame_only_after_consistent_observations(self):
        app = self.relocalization_app()
        for i in range(2):
            app._try_h_relocalization(10 + i * 0.1, 13, 12)
            self.assertFalse(app.pose.position_valid)
        app._try_h_relocalization(10.2, 13, 12)
        self.assertTrue(app.pose.position_valid)
        self.assertEqual(app.pose.get()[:2], (47, 18))

    def test_a_new_unknown_h_cannot_invent_the_original_coordinate_frame(self):
        app = self.relocalization_app()
        app.known_pad = None
        for i in range(4):
            app._try_h_relocalization(10 + i * 0.1, 13, 12)
        self.assertFalse(app.pose.position_valid)
        app.odometry.relocalize_from_pad.assert_not_called()

    def test_queued_or_just_dequeued_command_prevents_a_coordinate_shift(self):
        for dequeued in (False, True):
            with self.subTest(dequeued=dequeued):
                app = self.relocalization_app()
                app.commands.put({"type": "flip"})
                if dequeued:
                    app.commands.get_nowait()
                for i in range(4):
                    app._try_h_relocalization(10 + i * 0.1, 13, 12)
                self.assertFalse(app.pose.position_valid)
                app.odometry.relocalize_from_pad.assert_not_called()

    def test_inconsistent_h_observations_do_not_restore_global_coordinates(self):
        app = self.relocalization_app()
        for i, xy in enumerate(((13, 12), (30, -4), (0, 25))):
            app._try_h_relocalization(10 + i * 0.1, *xy)
        self.assertFalse(app.pose.position_valid)

    def test_unknown_global_position_does_not_place_false_terrain_blocks(self):
        telemetry = Telemetry.__new__(Telemetry)
        telemetry.pose = Pose(z=60)
        telemetry.pose.measured = True
        telemetry.pose.position_valid = False
        telemetry.pose.local_reference_valid = True
        telemetry.tello = Mock()
        telemetry.tello.get_current_state.return_value = {"tof": 60, "baro": 1.95}
        telemetry.is_airborne = lambda: True
        telemetry.terrain = TerrainMap()
        telemetry._baro0 = 0
        telemetry._alt = None
        for i in range(20):
            sample = telemetry._sample(i * 0.1, 0.1)
        self.assertEqual(sample["tof"], 60)
        self.assertEqual(sample["alt"], 195)  # retain the two distinct real measurements
        self.assertIsNone(sample["ground"])
        self.assertFalse(sample["position_valid"])
        self.assertEqual(telemetry.terrain.snapshot(), [])

    def test_rejected_camera_command_does_not_change_the_camera_label(self):
        app = DroneApp.__new__(DroneApp)
        app.downvision_enabled = False
        app.tello = Mock()
        app.tello.send_command_with_return.return_value = "error"
        with self.assertRaises(CommandRejected):
            app.set_downvision(True)
        self.assertFalse(app.downvision_enabled)

    def test_h_is_memorized_in_local_coordinates_without_inventing_a_global_fix(self):
        app = DroneApp.__new__(DroneApp)
        app.pose = Pose(z=80)
        app.pose.measured = True
        app.pose.position_valid = False
        app.pose.local_reference_valid = True
        app.cam = DownCam()
        app.pad_detector = HelipadDetector()
        app.pad_tracker = HelipadTracker()
        app.local_pads = []
        app.local_known_pad = app.known_pad = None
        app.pads = []
        app.airborne = True
        app.recording = False
        app.downvision_enabled = True
        app._vis_lock = threading.Lock()
        app.read_height = Mock(return_value=80)
        frame = image_fixtures.DetectorImageTests().scene()
        for i in range(5):
            app.process_frame(frame.copy(), 10 + i * 0.1)
        self.assertTrue(app.ever_seen_h)
        self.assertIsNotNone(app.seen_helipad(local=True))
        self.assertIsNone(app.seen_helipad())
        self.assertFalse(app.pose.position_valid)
        app.takeoff = Mock()
        app.follower = Mock()
        app.follower.hold_position.return_value = True
        app.land_on_helipad = Mock(return_value=True)
        app.safe_land = Mock()
        app.helipad_test()
        app.follower.hold_position.assert_called_once()
        self.assertTrue(app.follower.hold_position.call_args.kwargs["allow_local_reference"])
        app.follower.move_to.assert_not_called()
        app.safe_land.assert_called_once_with(helipad=False)


if __name__ == "__main__":
    unittest.main()
