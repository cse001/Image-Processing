#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 2018

@author: sharvil
"""

# Write a computer program capable of reducing the number of intensity levels in an image
# from 256 to 2 (in various integer powers of 2 i.e from 1 to 8). Try displaying all the lab2_images in
# one figure to compare the difference (use subplot utility if you are using matlab). When this
# is used with a given image the results are as below.

import cv2
import matplotlib.pyplot as plt


def reduce_intensity(image_matrix, factor):
    return (image_matrix//factor)*factor

image = cv2.imread("moon.tif", cv2.IMREAD_GRAYSCALE)

plt.figure(1)
for intensity_factor in range(8):
    plt.subplot(2, 4, intensity_factor+1)
    plt.imshow(reduce_intensity(image, 2**intensity_factor), 'gray')
    plt.title(str(intensity_factor+1) + " bits")

plt.show()
