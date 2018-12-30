#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug Mon 20 2018

@author: sharvil
"""

# Write a computer program capable of zooming and shrinking an image by pixel
# replication. Assume that the desired zoom/shrink factors are integers. Take any image
# and use your program to shrink the image by a factor of 10. Use your program to zoom
# the image back to the resolution of the original. Explain the reasons for their differences.

import cv2
import numpy as np


def shrink(image, factor):
    return image[::factor, ::factor]


def zoom(image, factor):
    image = np.repeat(image, factor, axis=1)
    image = np.repeat(image, factor, axis=0)
    return image
# def shrink_resolution(img, factor):
#     # Reduces resolution of an image by reducing and re enlarging it using the average of the neighbouring pixels
#     shrunk = cv2.resize(img, (0, 0), None, 1.0/factor, 1.0/factor, cv2.INTER_AREA)
#     return cv2.resize(shrunk, (0, 0), None, factor, factor, cv2.INTER_AREA)

image = cv2.imread("moon.tif", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", image)
image = shrink(image, 10)
cv2.imshow("Shrunk Image", image)
image = zoom(image, 10)
cv2.imshow("Zoomed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
