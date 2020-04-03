import cv2
from glob import glob
import numpy as np
from pprint import pprint
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop

IMG_PATH = "/home/nj/IMAGES/**/*.png"
IMG_LIST = glob(IMG_PATH,recursive=True)

for i,path in enumerate(IMG_LIST):
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img = img[5:img.shape[0] - 5, 5:img.shape[1] - 5]
    # img = cv2.pyrMeanShiftFiltering(img, 21, 51)

    img = cv2.copyMakeBorder(
        img,
        top=20,
        bottom=20,
        left=20,
        right=20,
        borderType=cv2.BORDER_ISOLATED,
        value=[255, 255, 255]
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, (7, 7), 0)
    thresh = cv2.dilate(thresh,(5,5))

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    photo_list = []

    area_list = []
    total_contours = 0
    area_cumsum    = 0
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_list.append(area)
        # print(area)

    total_contours = len(contours)
    area_cumsum = sum(area_list)
    max_area = np.max(area_list)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > (area_cumsum/total_contours):
            print(area)
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            # cv2.drawContours(img, [approx], 0, (125,125,125), 5)

            x,y,w,h = cv2.boundingRect(cnt)

            # rect = cv2.minAreaRect(cnt)
            # print(rect)
            # photo = crop_minAreaRect(img, rect)
            # print(photo)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            photo = img[y:y + h, x:x + w]

            photo_list.append(photo)

    for count, pic in enumerate(photo_list):
        cv2.imwrite("/home/nj/NJ/GitHub/img_scan_assistant/data/" + str(count) + ".jpg", pic)

    cv2.imwrite("/home/nj/NJ/GitHub/img_scan_assistant/data/threshold.png",thresh)
    break