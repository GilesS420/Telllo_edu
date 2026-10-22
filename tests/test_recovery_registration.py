"""Image-only recovery checks: no SDK connection, movement, or artifact writes."""

import unittest

import cv2
import numpy as np

from drone.registration import recover_translation


class RecoveryRegistrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(142)
        self.reference = cv2.GaussianBlur(rng.uniform(0, 255, (240, 320)).astype(np.float32),
                                          (3, 3), 0.6)

    def shifted(self, shift, scale=1.0, yaw=0.0):
        normalize = np.vstack((cv2.getRotationMatrix2D((159.5, 119.5), yaw, scale), (0, 0, 1)))
        translation = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], dtype=float)
        return cv2.warpAffine(self.reference, (np.linalg.inv(normalize) @ translation)[:2], (320, 240))

    def test_large_offset_uses_the_real_shared_rectangle(self):
        current = self.shifted((126, 25))
        original = current.copy()
        shift, quality, info = recover_translation(self.reference, current, 1.0, 0.0)
        self.assertIsNotNone(shift, info)
        np.testing.assert_allclose(shift, (126, 25), atol=1)
        self.assertGreaterEqual(quality, 0.2)
        self.assertGreater(info["correlation"], 0.9)
        self.assertGreaterEqual(info["overlap_fraction"], 0.35)
        np.testing.assert_array_equal(current, original)

    def test_expected_height_scale_and_yaw_are_normalized(self):
        shift, _, info = recover_translation(self.reference, self.shifted((110, 20), 1.2, 8), 1.2, 8)
        self.assertIsNotNone(shift, info)
        np.testing.assert_allclose(shift, (110, 20), atol=1)
        self.assertAlmostEqual(info["registration_scale"], 1.2, delta=0.03)

    def test_sensor_scale_residual_can_use_the_native_image(self):
        shift, _, info = recover_translation(self.reference, self.shifted((126, 25)), 94 / 85, 0)
        self.assertIsNotNone(shift, info)
        np.testing.assert_allclose(shift, (126, 25), atol=1)
        self.assertAlmostEqual(info["registration_scale"], 1, delta=0.01)

    def test_partial_foreground_occlusion_still_requires_image_agreement(self):
        current = self.shifted((126, 25))
        current[90:210, 100:165] = 10
        shift, _, info = recover_translation(self.reference, current, 94 / 85, 0)
        self.assertIsNotNone(shift, info)
        np.testing.assert_allclose(shift, (126, 25), atol=1)
        self.assertGreaterEqual(info["correlation"], 0.6)

    def test_unrelated_textured_images_are_rejected(self):
        unrelated = np.random.default_rng(77).uniform(0, 255, self.reference.shape).astype(np.float32)
        self.assertIsNone(recover_translation(self.reference, unrelated, 1.0, 0)[0])

    def test_blank_images_are_rejected(self):
        blank = np.full_like(self.reference, 120)
        self.assertIsNone(recover_translation(blank, blank, 1.0, 0)[0])

    def test_repeated_floor_pattern_does_not_authorize_a_position(self):
        y, x = np.indices(self.reference.shape)
        tiles = (((x // 8 + y // 8) % 2) * 180 + 30).astype(np.float32)
        current = np.roll(tiles, (19, 57), axis=(0, 1))
        self.assertIsNone(recover_translation(tiles, current, 1.0, 0)[0])

    def test_repeated_detailed_tiles_do_not_authorize_a_position(self):
        tile = np.random.default_rng(12).uniform(0, 255, (24, 32)).astype(np.float32)
        tiles = np.tile(tile, (10, 10))
        self.assertIsNone(recover_translation(tiles, np.roll(tiles, (31, 81), axis=(0, 1)), 1, 0)[0])

    def test_less_than_required_overlap_is_rejected(self):
        self.assertIsNone(recover_translation(self.reference, self.shifted((150, 90)), 1, 0)[0])

    def test_half_frame_translation_is_ambiguous(self):
        self.assertIsNone(recover_translation(self.reference, self.shifted((160, 0)), 1, 0)[0])

    def test_invalid_sensor_geometry_is_rejected(self):
        for scale, yaw in ((float("nan"), 0), (0, 0), (1, float("inf")), (1, 40)):
            self.assertIsNone(recover_translation(self.reference, self.reference, scale, yaw)[0])
