import numpy as np
import cv2

img = cv2.imread('images/ball.bmp', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', hsv)
ball_color_lower_bound = (4, 45, 55)
ball_color_upper_bound = (14, 255, 255)
mask = cv2.inRange(hsv, ball_color_lower_bound, ball_color_upper_bound)
cv2.imshow('Detected ball using mask', mask)

structuring_element = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, structuring_element, iterations=5)
structuring_element = np.ones((11, 11), np.uint8)
mask = cv2.erode(mask, structuring_element, iterations=2)
mask = cv2.dilate(mask, structuring_element, iterations=2)
cv2.imshow('circles', mask)
contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
for c in contours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
    cv2.imshow("Output Image", img)
    cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
