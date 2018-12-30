import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes

boundaries = []
out_boundaries = {}
in_boundaries = {}


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


def construct_circular_structuring_element(rad):
    n = rad * 2 + 1
    struct_element = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if (i - n // 2) ** 2 + (j - n // 2) ** 2 < rad ** 2:
                struct_element[i, j] = 255
    return struct_element


image = cv2.imread('images/test1.bmp', cv2.IMREAD_GRAYSCALE)

struct_element_1 = construct_circular_structuring_element(11)
structuring_element_2 = np.zeros((3, 3), dtype=np.uint8)
structuring_element_2[1, 0] = 255
structuring_element_2[0, 1] = 255
structuring_element_2[1, 1] = 255
structuring_element_2[1, 2] = 255
structuring_element_2[2, 1] = 255

ans = erosion(image, struct_element_1)

ans2 = binary_fill_holes(ans)

fig = plt.figure(figsize=(20, 12))
fig.add_subplot(1, 3, 1)
plt.axis('off')
plt.title('Original')
plt.imshow(image, cmap='gray')
fig.add_subplot(1, 3, 2)
plt.axis('off')
plt.title('After erosion')
plt.imshow(ans, cmap='gray')
fig.add_subplot(1, 3, 3)
plt.axis('off')
plt.title('After filling holes')
plt.imshow(ans2, cmap='gray')
plt.show()
