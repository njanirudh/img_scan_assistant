import cv2
import uuid
import numpy as np
from os import path

class ImageCropper:

    def __init__(self, config_path:str):
        self.scale_factor = 0.1
        self.edge_crop = 5
        self.border = 20
        self.photo_list = []

    def process_image(self, image:np.array) -> list[np.array]:
        resized_img = self.__preprocess_rs_image(image)
        image_list = self.__crop_rs_image(image, resized_img)

        return image_list

    def __preprocess_rs_image(self, image:np.array) -> np.array:
        rs_image = cv2.resize(image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        width,height,channels = rs_image.shape

        rs_image = rs_image[self.edge_crop:width - self.edge_crop,
                            self.edge_crop:height - self.edge_crop]
        rs_image = image[int((1 / self.scale_factor) * self.edge_crop): image[0] - int((1 / self.scale_factor) * self.edge_crop),
                   int((1 / self.scale_factor) * self.edge_crop):image[1] - int((1 / self.scale_factor) * self.edge_crop)]

        rs_image = cv2.pyrMeanShiftFiltering(rs_image, 21, 51)
        rs_image = cv2.cvtColor(rs_image, cv2.COLOR_BGR2GRAY)

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

    def __crop_rs_image(self, image:np.array, rs_image:np.array) -> list[np.array]:
        _,contours,_ = cv2.findContours(rs_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_list.append(area)

        total_contours = len(contours)
        area_cumsum = sum(area_list)
        max_area = np.max(area_list)

        photo_list:list = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > (area_cumsum / total_contours) and area != max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                x, y, w, h = int((1 / self.scale_factor) * (x - self.border)), int((1 / self.scale_factor) * (y - self.border)), \
                             int((1 / self.scale_factor) * (w)), int((1 / self.scale_factor) * (h))

                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                photo = image[y:y + h, x:x + w]

                photo_list.append(photo)
        
        return photo_list

