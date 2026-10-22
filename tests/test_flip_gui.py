import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gui.tracking import flip_readiness
from tello_gui import TelloGUI


class FlipReadinessTests(unittest.TestCase):
    def test_takeoff_tracking_flag_does_not_mean_the_sdk_is_ready(self):
        for state, phase in (("taking_off", None), ("flying", "takeoff_sdk")):
            with self.subTest(state=state, phase=phase):
                enabled, caption, reason = flip_readiness(state, True, phase, 90)
                self.assertFalse(enabled)
                self.assertIn("opstijgen", caption)
                self.assertIn("opstijgen", reason)

    def test_optional_recentering_can_be_interrupted_once_flying(self):
        enabled, _, _ = flip_readiness("flying", True, "takeoff_recenter", 90)
        self.assertTrue(enabled)

    def test_ground_landing_and_existing_flip_requests_are_unavailable(self):
        for state, airborne, busy in (("idle", False, False), ("landing", True, False),
                                      ("flying", True, True)):
            with self.subTest(state=state, airborne=airborne, busy=busy):
                enabled, _, reason = flip_readiness(state, airborne, battery=90,
                                                    flip_busy=busy)
                self.assertFalse(enabled)
                self.assertTrue(reason)

    def test_battery_warning_does_not_replace_the_configured_minimum(self):
        with patch("gui.tracking.cfg.FLIP_MIN_BATTERY", 20):
            self.assertFalse(flip_readiness("flying", True, battery=19)[0])
            for battery in (20, 35, 49):
                with self.subTest(battery=battery):
                    enabled, caption, warning = flip_readiness("flying", True, battery=battery)
                    self.assertTrue(enabled)
                    self.assertIn("<50%", caption)
                    self.assertIn("weigeren", warning)
            self.assertEqual(flip_readiness("flying", True, battery=50), (True, "🔁  Salto", ""))
            self.assertTrue(flip_readiness("flying", True, battery=None)[0])
        with patch("gui.tracking.cfg.FLIP_MIN_BATTERY", 70):
            self.assertFalse(flip_readiness("flying", True, battery=60)[0])


class FlipGUIRequestTests(unittest.TestCase):
    def setUp(self):
        self.gui = TelloGUI.__new__(TelloGUI)
        self.gui.app = SimpleNamespace(
            state="flying", airborne=True, _motion_phase=None, _active_command=None,
            commands=SimpleNamespace(flip_pending=threading.Event()),
            telemetry=SimpleNamespace(latest={"battery": 90}), submit=Mock(return_value=True))
        self.gui.flip_button = Mock()
        self.gui.toast = Mock()

    def test_programmatic_flip_cannot_bypass_takeoff_guard(self):
        self.gui.app._motion_phase = "takeoff_sdk"
        self.gui._flip()
        self.gui.app.submit.assert_not_called()
        self.assertIn("opstijgen", self.gui.toast.show.call_args.args[0])
        self.assertEqual(self.gui.flip_button.configure.call_args.kwargs["state"], "disabled")

    def test_rejected_submission_does_not_claim_a_flip_was_requested(self):
        self.gui.app.submit.return_value = False
        self.gui._flip()
        self.gui.app.submit.assert_called_once_with({"type": "flip", "dir": "f"})
        self.assertIn("niet ingepland", self.gui.toast.show.call_args.args[0])
        self.assertNotIn("aangevraagd", self.gui.toast.show.call_args.args[0])

    def test_successful_request_disables_the_button_before_the_next_tick(self):
        def submit(command):
            self.gui.app.commands.flip_pending.set()
            return True

        self.gui.app.submit.side_effect = submit
        self.gui._flip("b")
        self.gui.app.submit.assert_called_once_with({"type": "flip", "dir": "b"})
        self.assertEqual(self.gui.flip_button.configure.call_args.kwargs["state"], "disabled")
        self.gui.toast.show.assert_called_once_with("Salto aangevraagd", "info", 2)
        self.gui._flip()
        self.assertEqual(self.gui.app.submit.call_count, 1)

    def test_active_flip_stays_disabled_after_it_leaves_the_queue(self):
        self.gui.app._active_command = "flip"
        self.gui._refresh_flip_button()
        self.assertEqual(self.gui.flip_button.configure.call_args.kwargs["state"], "disabled")
        self.gui._flip()
        self.gui.app.submit.assert_not_called()

    def test_click_uses_current_battery_and_does_not_block_the_firmware_warning(self):
        self.gui._flip_battery = 90
        with patch("gui.tracking.cfg.FLIP_MIN_BATTERY", 20):
            self.gui.app.telemetry.latest["battery"] = 19
            self.gui._flip()
            self.gui.app.submit.assert_not_called()
            self.gui.app.telemetry.latest["battery"] = 30
            self.gui._flip()
        self.gui.app.submit.assert_called_once()
        self.assertIn("aangevraagd", self.gui.toast.show.call_args.args[0])
        self.assertIn("<50%", self.gui.toast.show.call_args.args[0])
        self.assertEqual(self.gui.toast.show.call_args.args[1], "warn")

    def test_topbar_refreshes_the_flip_caption_as_telemetry_changes(self):
        self.gui.state_pill = Mock()
        self.gui.battery = Mock()
        self.gui.top_vars = {key: Mock() for key in ("time", "height", "odo", "jetson")}
        self.gui.rec_pill = Mock()
        self.gui.rec_pill.winfo_ismapped.return_value = False
        self.gui.show = [0, 0, 80]
        self.gui.takeoff_time = None
        self.gui.app.pose = SimpleNamespace(
            snapshot=lambda: {"measured": True, "position_valid": True, "measurement_time": 10},
            quality=1)
        self.gui.app.link = SimpleNamespace(last_rx=None)
        self.gui.app.recording = False
        for battery in (90, 30):
            self.gui.app.telemetry.latest = {"battery": battery}
            self.gui._topbar(self.gui.app.telemetry.latest, 10)
        self.assertIn("<50%", self.gui.flip_button.configure.call_args.kwargs["text"])
        self.assertEqual(self.gui.flip_button.configure.call_args.kwargs["state"], "normal")


if __name__ == "__main__":
    unittest.main()
