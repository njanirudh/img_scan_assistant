import cv2
import unittest
import numpy as np
from unittest.mock import patch, mock_open
from image_scan_assistant import ImageCropper


class TestImageCropper(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="""
    preprocessing:
      scale_factor: 0.5
      edge_crop: 10
      border: 5
    """)
    def setUp(self, mock_file):
        """
        Set up the test case with a mock configuration.
        """
        self.cropper = ImageCropper(config="mock_config.yaml")

    def test_preprocess_rs_image(self):
        """
        Test if the image preprocessing step works correctly.
        """
        # Create a synthetic test image (RGB 100x100 white image)
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Call the preprocessing method
        processed_image = self.cropper._preprocess_rs_image(test_image)

        # Ensure the result is not None and is a binary image (single-channel)
        self.assertIsNotNone(processed_image)
        self.assertEqual(len(processed_image.shape), 2)  # Check if the image is single-channel
        self.assertEqual(processed_image.shape[0], 80)  # Ensure the image is cropped correctly (after edge crop)
        self.assertEqual(processed_image.shape[1], 80)

    def test_crop_rs_image(self):
        """
        Test the cropping function by creating a synthetic binary image with contours.
        """
        # Create a synthetic test image (RGB 100x100 white image)
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Create a binary image with a black rectangle in the center (mimicking a photo)
        binary_image = np.ones((100, 100), dtype=np.uint8) * 255  # White background
        cv2.rectangle(binary_image, (20, 20), (80, 80), 0, -1)  # Black rectangle as a "photo"

        # Call the cropping method
        cropped_images = self.cropper._crop_rs_image(test_image, binary_image)

        # Check that at least one cropped image is returned
        self.assertEqual(len(cropped_images), 1)

        # Ensure that the cropped image has the expected dimensions
        cropped_img = cropped_images[0]
        self.assertEqual(cropped_img.shape[0], 60)  # Height of the cropped image
        self.assertEqual(cropped_img.shape[1], 60)  # Width of the cropped image

    def test_process_image(self):
        """
        Test the entire process_image function.
        """
        # Create a synthetic test image (RGB 100x100 white image)
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Call the process_image method
        image_list, resized_img = self.cropper.process_image(test_image)

        # Ensure that the resized image is not None and has expected dimensions
        self.assertIsNotNone(resized_img)
        self.assertEqual(resized_img.shape[0], 45)  # Check height (scale factor applied)
        self.assertEqual(resized_img.shape[1], 45)  # Check width (scale factor applied)

        # Ensure image_list is returned (although it may be empty)
        self.assertIsInstance(image_list, list)

    def test_empty_image(self):
        """
        Test how the cropper handles an empty (all white) image.
        """
        # Create a synthetic test image (RGB 100x100 white image)
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Call the process_image method
        image_list, resized_img = self.cropper.process_image(test_image)

        # Check that no cropped images are returned for an empty image
        self.assertEqual(len(image_list), 0)

    def test_small_image(self):
        """
        Test how the cropper handles a very small image.
        """
        # Create a synthetic small test image (10x10 white image)
        small_image = np.ones((10, 10, 3), dtype=np.uint8) * 255

        # Call the process_image method
        image_list, resized_img = self.cropper.process_image(small_image)

        # Since the image is too small, no cropping should happen
        self.assertEqual(len(image_list), 0)

        # Ensure the resized image is not None (it will be scaled down even further)
        self.assertIsNotNone(resized_img)


if __name__ == "__main__":
    unittest.main()
