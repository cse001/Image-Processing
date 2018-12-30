import cv2

image = cv2.imread('images/ball.bmp', 1)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('Original image', image)
cv2.imshow('HSV ', hsv)
edges = cv2.Canny(hsv, 100, 200)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
