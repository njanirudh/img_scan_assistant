import os
import glob
import uuid
import itertools
from os import path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cv2
import numpy as np

from image_scan_assistant._image_cropper import ImageCropper


class MultiCropper:
    """
    Multi-threaded cropper that uses a thread-pool
    to perform async processing on a list of images.
    """

    def __init__(self, threads: int, config: str):
        self.threads = threads
        self.config = config

    def crop_image_folder(self, in_path: str, out_path: str) -> None:
        """
        Asynchronously crop all the scanned images found in a folder.
        """

        # Get the list of image files (jpg and png)
        file_list = list(itertools.chain.from_iterable([glob.glob(pth, recursive=True) for pth in
                                                        [os.path.join(in_path, '*.jpg'),
                                                         os.path.join(in_path, '*.png')]]))

        # Create the output directory if it doesn't exist
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Multithreaded cropping
        with ProcessPoolExecutor(max_workers=self.threads) as executor:
            image_cropper = ImageCropper(config=self.config)

            futures = {
                executor.submit(image_cropper.process_image, cv2.imread(li)): li for li in file_list
            }

            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]  # Retrieve the file path associated with the future
                try:
                    result_list, result_debug = future.result()
                    # Save the cropped images and the processed image
                    self._save_image(result_debug, out_path)
                    self._save_image_list(result_list, out_path)
                except Exception as exc:
                    print(f"Error processing {file_path}: {exc}")

    def crop_single_image(self, in_path: str, out_path: str) -> None:
        """
        Crop a single scanned image.
        """
        image = cv2.imread(in_path)
        image_cropper = ImageCropper(config=self.config)

        try:
            result_list, _ = image_cropper.process_image(image)
            self._save_image_list(result_list, out_path)
        except Exception as exc:
            print(f"Error processing {in_path}: {exc}")

    def _save_image_list(self, image_list: list, out_path: str) -> None:
        """
        Saves all cropped photographs into the given folder path with a random file name.
        """
        for img in image_list:
            out_img_path = path.join(out_path, str(uuid.uuid4()) + ".jpg")
            cv2.imwrite(out_img_path, img)

    def _save_image(self, image: np.array, out_path: str) -> None:
        """
        Saves a single image to a folder with a random file name.
        """
        out_img_path = path.join(out_path, str(uuid.uuid4()) + ".jpg")
        print(f"Saving image to {out_img_path}")
        cv2.imwrite(out_img_path, image)

