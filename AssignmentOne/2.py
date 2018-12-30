#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 2018

@author: sharvil
"""

# Write a program that calculates the average intensity value of the pixels in the image
# Moon.bmp and then thresholds this image based on this intensity. Thresholding means that
# a new image is generated in which each pixel has intensity 1.0 if the corresponding pixel in
# the original image has a value above the threshold and 0 otherwise. Use this new function
# from the command line on the image moon.tif and it should have the following effect:

import cv2

image = cv2.imread("moon.tif", cv2.IMREAD_GRAYSCALE)
retval, threshold_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

cv2.imshow("Original Image", image)
cv2.imshow("Thresholded Image", threshold_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


