import cv2
import glob
from src.image_cropper import ImageCropper
from pathos.multiprocessing import ProcessPool
from src.cropper_pipeline import CropperPipeline

def threaded_cropper(input_path,output_path):
    img_cropper = ImageCropper()
    img = cv2.imread(input_path)
    img_cropper.set_input_image(img)
    img_cropper.preprocess_input_image()
    img_cropper.crop_image()
    img_cropper.save_cropped_image(output_path)
    img_cropper.reset()

if __name__ == "__main__":
    dataset = "/home/anirudh/NJ/Github/img_scan_assistant/dataset"
    results = "/home/anirudh/NJ/Github/img_scan_assistant/results"

    file_list = glob.glob(dataset, recursive=True)

    pool = ProcessPool(nodes=4)
    pool.map(threaded_cropper,file_list,[results]*len(file_list))

