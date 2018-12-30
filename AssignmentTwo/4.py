import sys
import cv2
import numpy as np

# Using any programming language you feel comfortable with, load an image and then
# perform a simple spatial 3x3 average of image pixels. In other words, replace the value of
# every pixel by the average of the values in its 3x3 neighborhood. Be careful with pixels at
# the image boundaries. Repeat the process for a 10x10 neighbourhood and again for a
# 20x20 neighbourhood. Observe what happens to the image and give explanation for


def average_filter(orig, size):
    arr = np.zeros(orig.shape, dtype=np.uint8)
    avg_fil = np.ones([size, size])
    avg_fil = avg_fil / (size ** 2)
    rows = orig.shape[0]
    cols = orig.shape[1]
    pad = size // 2
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            arr[i][j] = np.sum(np.multiply(orig[i - pad: i + pad , j - pad: j + pad ], avg_fil))

    return arr


img = cv2.imread('lab2_images/bottom_left.tif', cv2.IMREAD_GRAYSCALE)
filter_size = int(sys.argv[1])
res = average_filter(img, filter_size)
cv2.imshow("Original", img)
cv2.imshow("After applying filter", res)
cv2.waitKey(10000)
cv2.destroyAllWindows()
