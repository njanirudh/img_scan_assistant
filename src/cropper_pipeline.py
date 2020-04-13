import cv2
import glob
from src.image_cropper import ImageCropper

class CropperPipeline:

    def __init__(self,threads = 0):
        self.threads = threads
        self.input_path = None
        self.output_path = None
        self.input_list = None

    def run_pipeline_on_folder(self,input_path:str,output_path_folder:str):
        file_list = glob.glob(input_path, recursive=True)
        img_cropper = ImageCropper()

        for single_file in file_list:
            img = cv2.imread(single_file)

            img_cropper.set_input_image(img)
            img_cropper.preprocess_image()
            img_cropper.crop_image()
            img_cropper.save_cropped_image(output_path_folder)
            img_cropper.reset()

    def run_pipeline_on_image(self,image_path:str,output_path:str):
        pass

if __name__ == "__main__":


