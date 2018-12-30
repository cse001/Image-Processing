#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug Mon 20 2018

@author: sharvil
"""

# Take an image and add to it random noise. Repeat this N times. Add the resulting lab2_images
# and take an average. What do you observe?

import cv2
import numpy as np
import sys
img = cv2.imread('moon.tif', cv2.IMREAD_GRAYSCALE)


def addnoise(img):
    arr = np.ndarray(img.shape, dtype=np.uint8)
    cv2.randn(arr, 0, 1)
    return img + arr


n = int(sys.argv[1])
arr = np.ndarray(img.shape, dtype=np.uint8)
cv2.imshow('Original', img)
imglist = []

while n > 0:
    imglist.append(img)
    img = addnoise(img)
    n = n - 1

res = np.zeros(img.shape, dtype=int)
for img in imglist:
    res = res + img

res = res // len(imglist)
cv2.imshow('Noisy image', imglist[n-1])
cv2.imshow(' averge  Noise', np.array(res, dtype=np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
