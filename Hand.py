import cv2
import numpy as np

# Load the image
img = cv2.imread(r'H:/NOW IN/open-hand-20912298.jpg')

# Convert image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define skin color range
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Apply skin color mask
mask = cv2.inRange(hsv, lower_skin, upper_skin)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (which is likely to be the hand)
max_contour = max(contours, key=cv2.contourArea)

# Draw the contour on the original image
cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 2)

# Display the image with the detected hand contour
cv2.imshow("Detected Hand", img)
cv2.waitKey(0)
