import cv2
import path
import uuid

class MultiCropper:
    def __init__(self):
        pass

    def crop_image_folder(self, path:str):
        pass

    def crop_single_image(self, image:str):
        pass

    def __save_image_list(self, out_path:str, image_list:list):
        out_path_list = []
        for img in image_list:
            out_img_path = path.join(out_path,str(uuid.uuid4())+".jpg")
            out_path_list.append(out_img_path)
            cv2.imwrite(out_img_path,img)
            return out_path_list

    def __save_image(self, image:np.array, out_path:str):
        out_img_path = path.join(path, str(uuid.uuid4())+".jpg")
        cv2.imwrite(out_img_path, image)

