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
        self.scale_factor = 0.1
        self.edge_crop = 5
        self.border = 20
        self.photo_list = []

    def set_input_image(self,img):
        """
        Set input scanned image.
        :param img: Input image for cropping
        :return: None
        """
        self.input_image = img
        self.preprocessed_image = img

    def preprocess_input_image(self):
        """
        Preprocess input image and generate thresholded image to
        find contours in the further step.
        :return: None
        """

        self.preprocessed_image = cv2.resize(self.preprocessed_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        # print(self.preprocessed_image.shape)
        width,height,channels = self.preprocessed_image.shape

        self.preprocessed_image = self.preprocessed_image[self.edge_crop:width - self.edge_crop,
                                  self.edge_crop:height - self.edge_crop]
        self.input_image = self.input_image[int((1 / self.scale_factor) * self.edge_crop):self.input_image.shape[0] - int((1 / self.scale_factor) * self.edge_crop),
                   int((1 / self.scale_factor) * self.edge_crop):self.input_image.shape[1] - int((1 / self.scale_factor) * self.edge_crop)]

        self.preprocessed_image = cv2.pyrMeanShiftFiltering(self.preprocessed_image, 21, 51)
        self.preprocessed_image = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2GRAY)

        self.preprocessed_image = cv2.copyMakeBorder(
            self.preprocessed_image,
            top=self.border,
            bottom=self.border,
            left=self.border,
            right=self.border,
            borderType=cv2.BORDER_ISOLATED,
            value=[255, 255, 255]
        )

        _, self.preprocessed_image = cv2.threshold(self.preprocessed_image, 235, 255, cv2.THRESH_BINARY)
        self.preprocessed_image = cv2.GaussianBlur(self.preprocessed_image, (7, 7), 0)
        self.preprocessed_image = cv2.dilate(self.preprocessed_image, (5, 5))

        return self.preprocessed_image

    def crop_image(self):
        """
        Crops full scanned image into individual images.
        :return: None
        """
        _,contours,_ = cv2.findContours(self.preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                x, y, w, h = int((1 / self.scale_factor) * (x - self.border)), int((1 / self.scale_factor) * (y - self.border)), \
                             int((1 / self.scale_factor) * (w)), int((1 / self.scale_factor) * (h))

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
            cv2.imwrite(path.join(out_path,str(uuid.uuid4())+".jpg"),img)

    def reset(self):
        """
        Reset all class variables.
        :return: None
        """
        self.input_image = None
        self.output_image = None
        self.preprocessed_image = None
        self.photo_list = []

if __name__ == "__main__":

    input_path = "/home/anirudh/NJ/Github/img_scan_assistant/dataset/imagespng-01.png"
    output_path = "/home/anirudh/NJ/Github/img_scan_assistant/results"
    img = cv2.imread(input_path)

    img_cropper = ImageCropper()
    img_cropper.set_input_image(img)
    img_cropper.preprocess_input_image()
    img_cropper.crop_image()
    img_cropper.save_cropped_image(output_path)
    img_cropper.reset()
