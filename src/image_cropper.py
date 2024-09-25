import yaml
from typing import List

import cv2
import numpy as np


class ImageCropper:

    def __init__(self, config: str = "../config/config.yaml"):
        """
        Creates a cropper from configuration values given
        in the config.yaml file.
        """

        # Open config file from path
        with open(config) as stream:
            try:
                yaml_data = yaml.safe_load(stream)
                self.scale_factor = yaml_data["preprocessing"]["scale_factor"]
                self.edge_crop = yaml_data["preprocessing"]["edge_crop"]
                self.border = yaml_data["preprocessing"]["border"]

            except yaml.YAMLError as exc:
                print(exc)

            finally:
                # Setting preprocessing parameters
                self.scale_factor = 0.1
                self.edge_crop = 5
                self.border = 15

        self.minimum_area = 1000

    def process_image(self, image: np.array) -> (List[np.array], np.array):
        """
        Main function called to find the photographs in the scanned output image.
        """

        resized_img = self._preprocess_rs_image(image)
        image_list = self._crop_rs_image(image, resized_img)

        return image_list, resized_img

    def _preprocess_rs_image(self, image: np.array) -> np.array:
        """
        Perform preprocess steps on the image to find the photographs
        and create a binary image for further contour detection.
        """

        rs_image = cv2.resize(image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        width, height, channels = rs_image.shape

        # Remove a thin border from the image that
        # usually causes bad image thresholding and cropping
        rs_image = rs_image[self.edge_crop:width - self.edge_crop,
                   self.edge_crop:height - self.edge_crop]

        rs_image = cv2.pyrMeanShiftFiltering(rs_image, 21, 51)
        rs_image = cv2.cvtColor(rs_image, cv2.COLOR_BGR2GRAY)

        # Add a white background around the processed image
        rs_image = cv2.copyMakeBorder(
            rs_image,
            top=self.border,
            bottom=self.border,
            left=self.border,
            right=self.border,
            borderType=cv2.BORDER_ISOLATED,
            value=[255, 255, 255]
        )

        _, rs_image = cv2.threshold(rs_image, 235, 255, cv2.THRESH_BINARY)
        rs_image = cv2.GaussianBlur(rs_image, (7, 7), 0)
        rs_image = cv2.dilate(rs_image, (5, 5))

        return rs_image

    def _crop_rs_image(self, image: np.array, rs_image: np.array) -> List[np.array]:
        """
        Finds the largest contours in the binary image and returns
        a list of photographs found in the scanned image.
        """
        contours, hierarchy = cv2.findContours(rs_image, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Use contour area as a metric
        # for removing unwanted contours
        area_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_list.append(area)

        max_area = np.max(area_list)
        photo_list = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.minimum_area and area != max_area:
                # Resize contour rectangles to full size
                x, y, w, h = cv2.boundingRect(cnt)
                x, y, w, h = int((1 / self.scale_factor) * (x - self.border)), \
                             int((1 / self.scale_factor) * (y - self.border)), \
                             int((1 / self.scale_factor) * w), int((1 / self.scale_factor) * h)

                photo = image[y:y + h, x:x + w]
                photo_list.append(photo)

        return photo_list
