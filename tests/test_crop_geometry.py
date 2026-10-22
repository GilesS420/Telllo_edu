import unittest
from unittest.mock import patch

import numpy as np

from drone import config as cfg
from drone.downcam import DownCam


class CropGeometryTests(unittest.TestCase):
    def test_one_dark_side_does_not_shift_the_optical_centre(self):
        camera = DownCam()
        image = np.full((240, 320), 180, np.uint8)
        image[:, :33] = 0
        camera._set_crop(image, image.shape)
        self.assertEqual(camera.crop, (0, 0, 320, 240))

    def test_uniform_balanced_encoded_padding_is_still_supported(self):
        camera = DownCam()
        image = np.zeros((240, 400), np.uint8)
        image[:, 40:360] = 180
        camera._set_crop(image, image.shape)
        self.assertEqual(camera.crop, (40, 0, 320, 240))

    def test_dark_textured_sides_are_not_encoded_padding(self):
        camera = DownCam()
        image = np.random.default_rng(1).integers(0, 10, (240, 400), dtype=np.uint8)
        image[:, 40:360] = 180
        camera._set_crop(image, image.shape)
        self.assertEqual(camera.crop, (0, 0, 400, 240))

    def test_disabled_auto_crop_keeps_full_frame_as_scene_changes(self):
        camera = DownCam()
        with patch.object(cfg, "AUTO_CROP", False), patch.object(cfg, "DOWNCAM_CROP", None):
            for i in range(15):
                image = np.full((240, 320), 180, np.uint8)
                image[:, :33 + i] = 0
                self.assertEqual(camera.prepare(image).shape, (240, 320))
