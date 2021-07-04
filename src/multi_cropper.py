import os
import cv2
import glob
import uuid
import itertools
import numpy as np
from os import path

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from image_cropper import ImageCropper


class MultiCropper:

    def __init__(self, threads: int, config: str):
        """

        """
        self.threads = threads
        self.config = config

    def crop_image_folder(self, in_path: str, out_path: str) -> None:
        """

        """
        file_list = list(itertools.chain.from_iterable([glob.glob(pth, recursive=True) for pth in
                                                        [os.path.join(in_path, '*.jpg'),
                                                        os.path.join(in_path, '*.png')]]))
        # print(file_list)
        image_cropper = ImageCropper(config=self.config)

        # Multi-threaded cropper
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = []
            for li in file_list:
                img = cv2.imread(li)
                futures.append(executor.submit(image_cropper.process_image, img))

            for future in concurrent.futures.as_completed(futures):
                # print(len(future.result()[0]))
                result_list, result_debug = future.result()[0], future.result()[1]
                self.__save_image(result_debug, out_path)
                self.__save_image_list(result_list, out_path)

    def crop_single_image(self, in_path: str, out_path: str) -> None:
        """

        """
        image = cv2.imread(in_path)
        image_cropper = ImageCropper(config=self.config)
        result_list, _ = image_cropper.process_image(image)

        self.__save_image_list(result_list, out_path)

    def __save_image_list(self, image_list: list, out_path: str) -> None:
        """

        """
        out_path_list = []
        for img in image_list:
            out_img_path = path.join(out_path, str(uuid.uuid4()) + ".jpg")
            out_path_list.append(out_img_path)
            cv2.imwrite(out_img_path, img)

    def __save_image(self, image: np.array, out_path: str) -> None:
        """

        """
        out_img_path = path.join(out_path, str(uuid.uuid4()) + ".jpg")
        print(out_img_path)
        cv2.imwrite(out_img_path, image)


if __name__ == "__main__":
    config_file = "/home/anirudh/NJ/Github/img_scan_assistant/config/config.yaml"
    dataset_path = "/home/anirudh/NJ/Github/img_scan_assistant/dataset/*.png"

    threaded_cropper = MultiCropper(threads=4, config=config_file)
    threaded_cropper.crop_image_folder(dataset_path, "./")
