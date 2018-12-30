import cv2
import numpy as np
import matplotlib.pyplot as plt


def erosion(img):
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


def dilation(img):
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


def opening(img):
    eroded_image = erosion(img)
    result = dilation(eroded_image)
    return result


def closing(img):
    temp = dilation(img)
    result = erosion(temp)
    return result


# src_image = 'images/body1.bmp'
src_image = 'images/body2.bmp'
image = cv2.imread(src_image, cv2.IMREAD_GRAYSCALE)

structuring_element = np.zeros((3, 3), dtype=np.uint8)
structuring_element[1, 0] = 255
structuring_element[0, 1] = 255
structuring_element[1, 1] = 255
structuring_element[1, 2] = 255
structuring_element[2, 1] = 255

ans = erosion(image)
fig = plt.figure(figsize=(20, 12))
fig.add_subplot(1, 2, 1)
plt.axis('off')
plt.imshow(image, cmap='gray')
fig.add_subplot(1, 2, 2)
plt.axis('off')
plt.imshow(ans, cmap='gray')
plt.show()
