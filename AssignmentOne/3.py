#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 2018

@author: sharvil
"""

# Write a program which generates the negative of an image. This means that a new image
# is created in which the pixel values are all equal to 1.0 minus the pixel value in the original
# image. When this is used with the image moon.tif the results are as follows:

import cv2


def negate_image(image_matrix):
    rows = len(image_matrix)
    cols = len(image_matrix[0])
    for row in range(rows):
        for col in range(cols):
            image_matrix[row][col] = 255 - image_matrix[row][col]

image = cv2.imread("moon.tif", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", image)
negate_image(image)
cv2.imshow("Negated Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
