import cv2
from matplotlib import pyplot as plt
from copy import deepcopy
from math import acos, sqrt, ceil

fig = plt.figure(figsize=(20, 20), facecolor="#777777")
img = cv2.imread('images/ball.bmp', cv2.IMREAD_COLOR)
resultant_images = []


def plot_figures(image_list, figure, title=None):
    length = len(image_list)
    square_root = sqrt(length)
    h = w = ceil(square_root)
    for ct, images in enumerate(image_list):
        figure.add_subplot(h, w, ct + 1)
        plt.axis('off')
        if title is not None:
            plt.title(title[ct])
        else:
            plt.title(ct)
        plt.imshow(images)


def separate_channels(image):
    c1 = deepcopy(image[:, :, :])
    c1[:, :, 1] = 0
    c1[:, :, 2] = 0
    c2 = deepcopy(image[:, :, :])
    c2[:, :, 0] = 0
    c2[:, :, 2] = 0
    c3 = deepcopy(image[:, :, :])
    c3[:, :, 0] = 0
    c3[:, :, 1] = 0
    return c1, c2, c3


def my_rgb_to_hsi(image):
    n, m = img.shape[0], img.shape[1]
    hsi_image = img.copy()

    for i in range(n):
        for j in range(m):
            r, g, b = image[i][j]
            r *= 1.0
            g *= 1.0
            b *= 1.0
            v = (r + g + b) / 3
            if v == 0:
                s = 0
            else:
                s = 1 - min(r, g, b) / v

            denominator = (2 * sqrt((r - g) * (r - g) + (r - b) * (g - b)))
            numerator = (r - g + r - b)
            h = 0
            if denominator == 0:
                theta = 0
            else:
                theta = acos(numerator / denominator)
            if b > g:
                h = theta
            else:
                h = 360 - theta
            h /= 360
            v /= 255

            h *= 255
            s *= 255
            v *= 255
            hsi_image[i][j] = h, s, v

    return hsi_image

image_two = my_rgb_to_hsi(img)
image_three = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
image_difference = image_three - image_two
temp = separate_channels(img)
img_temp_ls = [image_two, image_three, image_difference]
for i in img_temp_ls:
    temp = separate_channels(i)
    for j in temp:
        resultant_images.append(j)

titles = ["my_image_hue", "my_image_saturation", "my_image_intensity", "library_hue", "library_saturation", "library_intensity",
            "differnece_hue", "difference_saturation", "difference_intensity"]

plot_figures(resultant_images, fig, titles)
plt.show()
