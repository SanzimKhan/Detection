
import cv2
import numpy as np

img = cv2.imread(r'H:/NOW IN/licensed-image.jpg', cv2.IMREAD_GRAYSCALE)
img_blur = cv2.medianBlur(img, 5)
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=200, param2=20, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

cv2.imshow("Detected Circles", img)
cv2.waitKey(0)
