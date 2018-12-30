import cv2
import numpy as np
    
# Consider the figure below and the problem of matching a given image template (ex.
# Image with symbol ‘T’) to different regions/objects in a target image (the bigger
# image with many symbol). Based on the techniques studied till now, try
# implementing a method to locate the region in the target image which matches with
# the template image.


def find(image):
    rows, cols = image.shape
    tx, ty = (80, 80)
    sub_image_height, sub_image_width = (50, 50)
    sub_image = image[tx: tx + sub_image_height, ty: ty + sub_image_width]
    cv2.imshow("Sub Image", sub_image)

    sub_image_height, sub_image_width = sub_image.shape

    for h in range(0, rows - sub_image_height):
        found = False
        for w in range(0, cols - sub_image_width):
            if np.allclose(image[h: h + sub_image_height, w: w + sub_image_width], sub_image):
                image[h: h + sub_image_height, w: w + 1] = 255
                image[h: h + sub_image_height, w + sub_image_width - 1: w + sub_image_width] = 255
                image[h: h + 1, w: w + sub_image_width] = 255
                image[h + sub_image_height - 1: h + sub_image_height, w: w + sub_image_width] = 255
                cv2.imshow("Processed image", image)
                found = True
                break
        if found:
            break

img1 = cv2.imread("lab2_images/q6.png", 0)
cv2.imshow("Original Image", img1)
find(img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
