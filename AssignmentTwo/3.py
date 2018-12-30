import sys
import cv2
import numpy as np

# Implement a histogram equalization function. Use it to enhance the above images.
# Compare the output of your implementation with any built-in library function.


def generate_histogram(orig):
    hist = [0] * 256
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            hist[orig[i][j]] = hist[orig[i][j]] + 1
    return hist


def enhance_by_hist_equalisation(orig, hist):
    prev = 0
    transformation_function = [0] * 256
    for i in range(256):
        transformation_function[i] = int(prev + (255 / float(orig.shape[0] * orig.shape[1])) * hist[i])
        prev = prev + (255 / float(orig.shape[0] * orig.shape[1])) * hist[i]
    arr = np.ndarray(orig.shape, dtype=np.uint8)
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            arr[i][j] = transformation_function[orig[i][j]]
    return arr


c = int(sys.argv[1])
if c == 1:
    img = cv2.imread('lab2_images/top_left.tif', cv2.IMREAD_GRAYSCALE)
elif c == 2:
    img = cv2.imread('lab2_images/bottom_left.tif', cv2.IMREAD_GRAYSCALE)
else:
    img = cv2.imread('lab2_images/2nd_from_top.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", img)
histogram = generate_histogram(img)
eq_img = enhance_by_hist_equalisation(img, histogram)
cv2.imshow("Histogram Equalized Image", eq_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
