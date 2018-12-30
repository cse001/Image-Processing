#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 2018

@author: sharvil
"""

import sys
import numpy as np
import cv2

#   Write a function flipImage which flips an image either vertically or horizontally. The function
#   should take two parameters – the matrix storing the image data and a flag to indicate
#   whether the image should be flipped vertically or horizontally. Use this function from the
#   command line to flip a given image both vertically and horizontally which should give the
#   following results.


# def flip_image(image_matrix, orientation):
#     cv2.imshow("Original", image_matrix)
#     image_matrix = np.flip(image_matrix, orientation)
#     cv2.imshow("Rotated", image_matrix)

def flip_image(image_matrix, orientation):
    rows = len(image_matrix)
    cols = len(image_matrix[0])
    if orientation == 0:
        half_cols = cols/2
        for row in range(rows):
            for col in range(int(half_cols)):
                image_matrix[row][col], image_matrix[row][cols - col - 1] = image_matrix[row][cols - col - 1], image_matrix[row][col]
    else:
        half_rows = rows/2
        for row in range(int(half_rows)):
            for col in range(cols):
                image_matrix[row][col], image_matrix[rows - row - 1][col] = image_matrix[rows - row - 1][col], image_matrix[row][col]

try:
    flag = int(sys.argv[1])
except:
    flag = 0

image = cv2.imread("moon.tif", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", image)
flip_image(image, flag)
cv2.imshow("Flipped Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
