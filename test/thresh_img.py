import  cv2
from PIL import Image

IMG_PATH = "/home/anirudh/Pictures/IMAGES/Other/1.png"
font = cv2.FONT_HERSHEY_COMPLEX

BORDER = 10
SUB=5
SCALE = 0.2

img = cv2.imread(IMG_PATH)
full_img = img

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(0,0),fx=SCALE,fy=SCALE)

img = img[SUB:img.shape[0]-SUB,SUB:img.shape[1]-SUB]
full_img = full_img[int((1/SCALE)*SUB):full_img.shape[0]-int((1/SCALE)*SUB),
           int((1/SCALE)*SUB):full_img.shape[1]-int((1/SCALE)*SUB)]

img = cv2.copyMakeBorder(
    img,
    top=BORDER,
    bottom=BORDER,
    left=BORDER,
    right=BORDER,
    borderType=cv2.BORDER_ISOLATED,
    value=[255, 255, 255]
)

_,thresh = cv2.threshold(img,225,255,cv2.THRESH_BINARY)

blur_img = cv2.GaussianBlur(thresh,(7,7),0)
# img_ccc = cv2.Canny(blur_img,50,150)

print(thresh.shape)
_,contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

photo_list = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000 and area < (img.shape[0]*img.shape[1]):
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (125,125,125), 5)
        x,y,w,h = cv2.boundingRect(cnt)
        # x,y,w,h = int((1/SCALE)*(x-BORDER+SUB)),int((1/SCALE)*(y-BORDER+SUB)),\
        #           int((1/SCALE)*(w-SUB)),int((1/SCALE)*(h-SUB))
        x,y,w,h = int((1/SCALE)*(x-BORDER)),int((1/SCALE)*(y-BORDER)),\
                  int((1/SCALE)*(w)),int((1/SCALE)*(h))

        cv2.rectangle(full_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        photo = full_img[y:y+h,x:x+w]

        photo_list.append(photo)

# for count,pic in enumerate(photo_list):
#     cv2.imwrite("/home/anirudh/NJ/Github/img_scan_assistant/dataset/"+str(count)+".jpg",pic)

# cv2.imshow("1", photo_list.pop(2))
cv2.imshow("2", thresh)
cv2.imwrite("/home/anirudh/NJ/Github/img_scan_assistant/dataset/"+"full"+".jpg",full_img)

cv2.waitKey(0)
