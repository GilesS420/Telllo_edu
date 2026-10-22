import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from drone import config as cfg
from drone.app import DroneApp
from drone.helipad import Helipad, HelipadDetector, HelipadTracker


def pad(x=100, y=100, score=0.2, size=50):
    return Helipad(x, y, size, score, np.zeros((4, 1, 2), np.int32))


class DetectorImageTests(unittest.TestCase):
    def scene(self, angle=0, inverse=False, broken=False):
        image = np.full((240, 320), 220, np.uint8)
        cv2.rectangle(image, (125, 85), (140, 155), 20, -1)
        cv2.rectangle(image, (180, 85), (195, 155), 20, -1)
        cv2.rectangle(image, (125, 112), (195, 128), 20, -1)
        if broken:
            image[97:99, 125:196] = 220  # narrow video/compression breaks in both bars
        image = cv2.warpAffine(image, cv2.getRotationMatrix2D((160, 120), angle, 1),
                               (320, 240), borderValue=220)
        if inverse:
            image = 255 - image
        noise = np.random.default_rng(5).normal(0, 3, image.shape)
        lighting = np.linspace(-25, 25, 320)[None, :]
        image = np.clip(image.astype(float) + noise + lighting, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(image, (3, 3), 0.7)

    def test_rotated_h_under_noise_and_uneven_lighting(self):
        detector = HelipadDetector()
        for inverse in (False, True):
            for angle in (0, 25, 70, 110):
                with self.subTest(angle=angle, inverse=inverse):
                    found = detector.find(self.scene(angle, inverse), 80)
                    self.assertIsNotNone(found)
                    self.assertLess(np.hypot(found.cx - 160, found.cy - 120), 3)

    def test_narrow_breaks_in_h_are_repaired(self):
        found = HelipadDetector().find(self.scene(broken=True), 80)
        self.assertIsNotNone(found)

    def test_solid_box_and_blank_floor_are_not_h(self):
        for box in (False, True):
            image = np.full((240, 320), 220, np.uint8)
            if box:
                cv2.rectangle(image, (125, 85), (195, 155), 20, -1)
            self.assertIsNone(HelipadDetector().find(image, 80))


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = HelipadTracker()
        self.shape = (240, 320)

    def confirm(self):
        for i in range(cfg.HELIPAD_MIN_SIGHTINGS):
            self.tracker.update([pad()], 10 + i * 0.1, self.shape)
        self.assertTrue(self.tracker.confirmed)

    def test_one_frame_cannot_confirm_a_landing_target(self):
        self.assertIsNone(self.tracker.update([pad()], 10, self.shape))
        self.assertFalse(self.tracker.confirmed)

    def test_repeated_timestamp_does_not_count_as_new_evidence(self):
        for _ in range(10):
            self.tracker.update([pad()], 10, self.shape)
        self.assertEqual(self.tracker.hits, 1)
        self.assertFalse(self.tracker.confirmed)

    def test_better_scoring_distractor_does_not_steal_a_locked_target(self):
        self.confirm()
        found = self.tracker.update([pad(250, score=0.01), pad(102)], 10.3, self.shape)
        self.assertLess(abs(found.cx - 102), 2)

    def test_low_height_motion_across_a_delayed_frame_keeps_the_target(self):
        for height, shift in ((60, 16), (40, 25)):
            with self.subTest(height=height, shift=shift):
                tracker = HelipadTracker()
                for i in range(cfg.HELIPAD_MIN_SIGHTINGS):
                    tracker.update([pad(size=32)], 10 + i * 0.1, self.shape,
                                   height_cm=height)
                timestamp = tracker.last_seen + 0.2
                found = tracker.update([pad(250, score=0.01, size=32),
                                        pad(100 + shift, size=32)],
                                       timestamp, self.shape, height_cm=height)
                self.assertIsNotNone(found)
                self.assertEqual(tracker.last_seen, timestamp)
                self.assertEqual(tracker.last_raw, (100 + shift, 100))
                self.assertTrue(tracker.confirmed)

    def test_height_gate_does_not_allow_the_same_pixel_jump_high_above_the_pad(self):
        for i in range(cfg.HELIPAD_MIN_SIGHTINGS):
            self.tracker.update([pad(size=32)], 10 + i * 0.1, self.shape,
                                height_cm=200)
        last_seen = self.tracker.last_seen
        self.tracker.update([pad(125, size=32)], last_seen + 0.2, self.shape,
                            height_cm=200)
        self.assertEqual(self.tracker.last_seen, last_seen)

    def test_low_height_motion_allowance_is_bounded(self):
        for i in range(cfg.HELIPAD_MIN_SIGHTINGS):
            self.tracker.update([pad(size=32)], 10 + i * 0.1, self.shape,
                                height_cm=11)
        last_seen = self.tracker.last_seen
        self.tracker.update([pad(170, size=32)], last_seen + 0.3, self.shape,
                            height_cm=11)
        self.assertEqual(self.tracker.last_seen, last_seen)

    def test_invalid_height_does_not_expand_the_motion_gate(self):
        for height in (None, 0, float("nan"), float("inf")):
            with self.subTest(height=height):
                tracker = HelipadTracker()
                for i in range(cfg.HELIPAD_MIN_SIGHTINGS):
                    tracker.update([pad(size=32)], 10 + i * 0.1, self.shape,
                                   height_cm=height)
                last_seen = tracker.last_seen
                tracker.update([pad(125, size=32)], last_seen + 0.2, self.shape,
                               height_cm=height)
                self.assertEqual(tracker.last_seen, last_seen)

    def test_brief_miss_keeps_overlay_without_refreshing_measurement(self):
        self.confirm()
        last_seen = self.tracker.last_seen
        self.assertIsNotNone(self.tracker.update([], last_seen + 0.1, self.shape))
        self.assertEqual(self.tracker.last_seen, last_seen)
        self.assertIsNone(self.tracker.update([], last_seen + cfg.HELIPAD_TRACK_HOLD_S + 0.01,
                                              self.shape))

    def test_large_jump_requires_a_new_confirmation(self):
        self.confirm()
        seen = self.tracker.last_seen
        self.tracker.update([pad(250)], seen + 0.1, self.shape)
        self.assertEqual(self.tracker.last_seen, seen)
        self.assertIsNone(self.tracker.update([pad(250)], seen + 0.5, self.shape))
        self.assertFalse(self.tracker.confirmed)

    def test_camera_shape_change_clears_the_old_track(self):
        self.confirm()
        self.assertIsNone(self.tracker.update([pad()], 10.3, (480, 640)))
        self.assertFalse(self.tracker.confirmed)

    def test_detect_loop_does_not_reprocess_a_frozen_video_frame(self):
        app = DroneApp.__new__(DroneApp)
        app.running = True
        app.frame_reader = Mock(frame=np.zeros(self.shape, np.uint8))
        app.process_frame = Mock()
        ticks = []

        def sleep(_):
            ticks.append(1)
            app.running = len(ticks) < 3

        with patch("drone.app.time.sleep", side_effect=sleep):
            app.detect_loop()
        app.process_frame.assert_called_once()


if __name__ == "__main__":
    unittest.main()
