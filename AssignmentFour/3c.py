import cv2
import numpy as np
import matplotlib.pyplot as plt


def erosion(img, structuring_element):
    rows, cols = img.shape
    k = structuring_element.shape[0]
    eroded_image = np.zeros(img.shape, dtype=np.uint8)
    for i in range(rows - k - 1):
        for j in range(cols - k - 1):
            sub_image = img[i:i + k, j:j + k]
            temp = np.multiply(sub_image, structuring_element)
            temp = temp * 255
            if (temp == structuring_element).all():
                eroded_image[i + k // 2, j + k // 2] = 255
    return eroded_image


def dilation(img, structuring_element):
    rows, cols = img.shape
    k = structuring_element.shape[0]
    dilated_image = np.zeros(img.shape, dtype=np.uint8)
    for i in range(rows - k - 1):
        for j in range(cols - k - 1):
            sub_image = img[i:i + k, j:j + k]
            temp = np.multiply(sub_image, structuring_element)
            temp = temp * 254
            temp = temp + 1
            if (temp == structuring_element).any():
                dilated_image[i + k // 2, j + k // 2] = 255
    return dilated_image


def opening(img, structuring_element):
    eroded_image = erosion(img, structuring_element)
    result = dilation(eroded_image, structuring_element)
    return result


def construct_circular_structuring_element(rad):
    n = rad * 2 + 1
    el = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if (i - n // 2) ** 2 + (j - n // 2) ** 2 < rad ** 2:
                el[i, j] = 255
    return el


image = cv2.imread('images/test 2.bmp', cv2.IMREAD_GRAYSCALE)

structuring_element = construct_circular_structuring_element(4)
ans = opening(image, structuring_element)
structuring_element = construct_circular_structuring_element(6)
ans = dilation(ans, structuring_element)

fig = plt.figure(figsize=(20, 12))
fig.add_subplot(1, 2, 1)
plt.axis('off')
plt.title('Original')
plt.imshow(image, cmap='gray')
fig.add_subplot(1, 2, 2)
plt.axis('off')
plt.title('After opening')
plt.imshow(ans, cmap='gray')
plt.show()
