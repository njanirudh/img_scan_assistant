import cv2
from glob import glob
import numpy as np
from pprint import pprint
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils

IMG_PATH = "/home/nj/IMAGES/**/*.png"
IMG_LIST = glob(IMG_PATH,recursive=True)

for i,path in enumerate(IMG_LIST):
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
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
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.GaussianBlur(thresh, (7, 7), 0)
    thresh = cv2.dilate(thresh,(5,5))

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]


    cv2.imshow("Output", img)
    cv2.waitKey(0)

    # cv2.imwrite("/home/nj/NJ/GitHub/img_scan_assistant/data/"+str(i)+".png",thresh)

    break