import cv2
import numpy as np
import matplotlib.pyplot as plt

resultant_image_one_set = set()
resultant_image_two_set = set()


def erosion(img, struct_element, which_set):
    rows, cols = img.shape
    k = struct_element.shape[0]
    res = np.zeros(img.shape, dtype=np.uint8)
    for i in range(rows - k - 1):
        for j in range(cols - k - 1):
            cut = img[i:i + k, j:j + k]
            temp = np.multiply(cut, struct_element)
            temp = temp * 255
            if (temp == struct_element).all():
                x = i + k // 2
                y = j + k // 2
                res[x, y] = 255
                if which_set is 0:
                    resultant_image_one_set.add((x, y))
                else:
                    resultant_image_two_set.add((x, y))
    return res

image = cv2.imread('images/semafor.bmp', cv2.IMREAD_GRAYSCALE)

structuring_element = np.ones((11, 11), dtype=np.uint8)
image_complement = 255 - image
resultant_image_one = erosion(image, structuring_element, 0)

window = np.zeros((13, 13), dtype=np.uint8)
for j in range(13):
    window[0, j] = 255
    window[12, j] = 255
    window[j, 0] = 255
    window[j, 12] = 255

resultant_image_two = erosion(image_complement, window, 1)

res = resultant_image_one * resultant_image_two
res = res * 255

ans = resultant_image_one_set.intersection(resultant_image_two_set)

fig = plt.figure(figsize=(20, 12))
fig.add_subplot(1, 3, 1)
plt.axis('off')
plt.title("resultant_image_one")
plt.imshow(resultant_image_one, cmap='gray')
fig.add_subplot(1, 3, 2)
plt.axis('off')
plt.title("resultant_image_two")
plt.imshow(resultant_image_two, cmap='gray')
fig.add_subplot(1, 3, 3)
plt.axis('off')
plt.title("Final Result")
plt.imshow(res, cmap='gray')
plt.show()
