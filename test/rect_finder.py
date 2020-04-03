import  cv2

IMG_PATH = "/home/nj/Documents/Family/Photos/1.png"
font = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread(IMG_PATH,0)
img = cv2.resize(img,(0,0),fx=0.1,fy=0.1)

# img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

img_ccc = cv2.Canny(img,50,150)

img_g1 = cv2.GaussianBlur(img,(9,9),0)
img_g2 = cv2.GaussianBlur(img,(3,3),0)
img_subtraction = img_g2 - img_g1

cv2.imshow("1",img_ccc)
cv2.imshow("2",img_subtraction)

# img_ccc = cv2.GaussianBlur(img_ccc,(5,5),0)

contours, _ = cv2.findContours(img_ccc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    print(area)

    if area > 200 :
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), font, 1, (0))
        elif len(approx) == 4:
            cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
        elif 6 < len(approx) < 15:
            cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
        else:
            cv2.putText(img, "Circle", (x, y), font, 1, (0))

cv2.imshow("3", img)
cv2.waitKey(0)