import  cv2
from PIL import Image

IMG_PATH = "/home/nj/Documents/Family/Photos/1.png"
font = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread(IMG_PATH,0)
img = cv2.resize(img,(0,0),fx=0.1,fy=0.1)
img = img[5:img.shape[0]-5,5:img.shape[1]-5]

img = cv2.copyMakeBorder(
    img,
    top=20,
    bottom=20,
    left=20,
    right=20,
    borderType=cv2.BORDER_ISOLATED,
    value=[255, 255, 255]
)

_,thresh = cv2.threshold(img,235,255,cv2.THRESH_BINARY)

blur_img = cv2.GaussianBlur(thresh,(7,7),0)
img_ccc = cv2.Canny(blur_img,50,150)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

photo_list = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000 and area < (img.shape[0]*img.shape[1]):
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # cv2.drawContours(img, [approx], 0, (125,125,125), 5)
        x,y,w,h = cv2.boundingRect(cnt)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        photo = img[y:y+h,x:x+w]

        photo_list.append(photo)

# for count,pic in enumerate(photo_list):
#     cv2.imwrite("/home/nj/NJ/GitHub/image_splitter/data/result_images"+str(count)+".jpg",pic)

# cv2.imshow("1", photo_list.pop(2))
cv2.imshow("2", thresh)

cv2.waitKey(0)
