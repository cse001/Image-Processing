import sys
import cv2
import numpy as np
import math

# For the attached image ‘spine.tiff’, enhance it using
# (a) The log transformation and (b) A power-law transformation
# In (a) the only free parameter is c, but in (b) there are two parameters, c and r for which
# values have to be selected. By experimentation, obtain the best visual enhancement
# possible with the methods in (a) and (b). Once (according to your judgment) you have the
# best visual result for each transformation, explain the reasons for the major differences
# between them.


def enhance_image(orig, transformation_type):
    arr = np.ndarray(img.shape, dtype=np.uint8)

    if transformation_type == 0:
        c = float(sys.argv[2])
        for i in range(orig.shape[0]):
            for j in range(orig.shape[1]):
                arr[i][j] = int(255 * c * math.log((1 + float(orig[i][j]) / 255), math.e))
    elif transformation_type == 1:
        c = float(sys.argv[2])
        r = float(sys.argv[3])
        for i in range(orig.shape[0]):
            for j in range(orig.shape[1]):
                arr[i][j] = int(255 * c * math.pow(float(orig[i][j] / 255), r))
    return arr


img = cv2.imread('lab2_images/fractured_spine.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", img)
transformation = int(sys.argv[1])
enhanced_image = enhance_image(img, transformation)

cv2.imshow("Enhanced", enhanced_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
