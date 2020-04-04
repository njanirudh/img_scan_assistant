import cv2
import numpy as np
from os import path
import uuid

class ImageCropper:
    """
    Class to pre-process the scanned image and crop
    individual images found.
    """

    def __init__(self):
        """
        Initialize variables for storing
        the image and output.
        """
        self.input_image  = None
        self.output_image = None
        self.preprocessed_image = None
        self.scale_factor = 0.5
        self.photo_list = []

    def set_input_image(self,img):
        """
        Set input scanned image.
        :param img: Input image for cropping
        :return: None
        """
        self.input_image = img

    def preprocess_image(self):
        """
        Preprocess input image and generate thresholded image to
        find contours in the further step.
        :return: None
        """
        self.preprocessed_image = cv2.resize(self.input_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        width,height = self.preprocessed_image.shape

        self.preprocessed_image = self.preprocessed_image[5:width - 5, 5:height - 5]
        self.preprocessed_image = cv2.pyrMeanShiftFiltering(self.preprocessed_image, 21, 51)

        self.preprocessed_image = cv2.copyMakeBorder(
            self.preprocessed_image,
            top=20,
            bottom=20,
            left=20,
            right=20,
            borderType=cv2.BORDER_ISOLATED,
            value=[255, 255, 255]
        )

        self.preprocessed_image = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2GRAY)
        _, self.preprocessed_image = cv2.threshold(self.preprocessed_image, 235, 255, cv2.THRESH_BINARY)
        self.preprocessed_image = cv2.GaussianBlur(self.preprocessed_image, (7, 7), 0)
        self.preprocessed_image = cv2.dilate(self.preprocessed_image, (5, 5))

        return self.preprocessed_image

    def crop_image(self):
        """
        Crops full scanned image into individual images.
        :return: None
        """
        contours, _ = cv2.findContours(self.preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_list.append(area)

        total_contours = len(contours)
        area_cumsum = sum(area_list)
        max_area = np.max(area_list)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > (area_cumsum / total_contours) and area != max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                photo = self.input_image[y:y + h, x:x + w]

                self.photo_list.append(photo)

    def get_output_image(self):
        """
        Return final output list of images.
        :return: List of images
        """
        self.photo_list

    def save_cropped_image(self,out_path:str):
        """
        Output path to save cropped images
        :param out_path: Folder path to save images
        :return: None
        """
        for img in self.photo_list:
            cv2.imwrite(path.join(out_path,uuid.uuid4()+".jpg"),img)

    def reset(self):
        """
        Reset all class variables.
        :return: None
        """
        self.input_image = None
        self.output_image = None
        self.preprocessed_image = None
        self.photo_list = []


