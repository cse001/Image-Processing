#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug Mon 20 2018

@author: sharvil
"""

# Use Matlab function to rotate any given image by 45 and 90 degrees. Try using different
# interpolation methods and see if there is any perceptible effect.

import cv2
import matplotlib.pyplot as plt


def rotate_image():
    image = cv2.imread('caat.jpg', cv2.IMREAD_GRAYSCALE)
    cols, rows = image.shape
    angle = 45
    plt.subplot(2, 2, 4)
    plt.imshow(image, 'gray')
    plt.title("Original")
    rot_mat = cv2.getRotationMatrix2D(((rows-1)/2, (cols-1)/2), angle, 1.0)
    plt.subplot(2, 2, 1)
    result = cv2.warpAffine(image, rot_mat, (rows, cols), flags=cv2.INTER_LINEAR)
    plt.imshow(result, 'gray')
    plt.title("Inter linear Image")
    plt.subplot(2, 2, 2)
    result = cv2.warpAffine(image, rot_mat, (rows, cols), flags=cv2.INTER_NEAREST)
    plt.imshow(result, 'gray')
    plt.title("Inter Nearest Image")
    plt.subplot(2, 2, 3)
    result = cv2.warpAffine(image, rot_mat, (rows, cols), flags=cv2.INTER_CUBIC)
    plt.imshow(result, 'gray')
    plt.title("Inter Cubic Image")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rotate_image()
