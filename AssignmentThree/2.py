"""
Create a spatial filter to get the horizontal gradient of the image “two_cats.jpg”. Create a
spatial filter to get the vertical gradient of the image (read the MATLAB documentation
of fspecial). Now transform both of these filters to the frequency domain. Also Transform
the two_cats image to the frequency domain. Apply the appropriate operations in the
frequency domain. Transform the data back into the spatial domain. Sum the horizontal
and vertical gradients components together. The resulting image should look like this:
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])


# def compute_dft(image_matrix, image_dft=None):
#     if image_dft is not None:
#         print(image_dft.shape)
#         fft = np.fft.fft2(image_matrix, s=image_dft.shape),
#     else:
#         fft = np.fft.fft2(image_matrix)
#
#     # fft_shift = np.fft.fftshift(fft)
#     return fft
#
#
# def plot_dft(fft_shift, title):
#     img_fft = np.log(np.abs(fft_shift) + 1)
#     plt.subplot(111), plt.imshow(img_fft, cmap='gray')
#     plt.title(title), plt.xticks([]), plt.yticks([])
#     plt.show()
#
#
# def compute_idft(freq_matrix):
#     res1 = np.fft.fft2(freq_matrix)
#     img_back = np.fft.ifft2(res1)
#     img_back = np.abs(img_back)
#     return img_back
#

# def normalise(arr, min_pixel_value, max_pixel_value):
#     rows, cols = arr.shape
#     for i in range(0, rows):
#         for j in range(1, cols - 1):
#             arr[i, j] = ((arr[i, j] - min_pixel_value) * 255) / (max_pixel_value - min_pixel_value)
#
#     return arr

#
# def apply_horizontal_gradient(img):
#     rows, cols = img.shape
#     arr = np.array(img, dtype=np.int16)
#     min_pixel_value = 256
#     max_pixel_value = -256
#     for i in range(0, rows):
#         for j in range(1, cols - 1):
#             arr[i, j] = img[i, j + 1] - img[i, j - 1]
#             min_pixel_value = min(min_pixel_value, arr[i, j])
#             max_pixel_value = max(max_pixel_value, arr[i, j])
#
#     normalised_arr = normalise(arr, min_pixel_value, max_pixel_value)
#     return normalised_arr
#
#
# def apply_vertical_gradient(img):
#     rows, cols = img.shape
#     arr = np.array(img)
#     min_pixel_value = 256
#     max_pixel_value = 0
#     for i in range(1, rows - 1):
#         for j in range(0, cols):
#             arr[i, j] = img[i + 1, j] - img[i - 1, j]
#             min_pixel_value = min(min_pixel_value, arr[i, j])
#             max_pixel_value = max(max_pixel_value, arr[i, j])
#
#     normalised_arr = normalise(arr, min_pixel_value, max_pixel_value)
#     return normalised_arr


img = cv2.imread('lab 3/two_cats.jpg', cv2.IMREAD_GRAYSCALE)

# img_dft = compute_dft(img)
img_dft = np.fft.fft2(img)
sobelx_filter_dft = np.fft.fft2(sobel_x, s=img_dft.shape)
sobely_filter_dft = np.fft.fft2(sobel_y, s=img_dft.shape)
filteredx_img_dft = img_dft*sobelx_filter_dft
filteredy_img_dft = img_dft*sobely_filter_dft

filteredx_img_back = np.fft.ifft2(filteredx_img_dft)
filteredy_img_back = np.fft.ifft2(filteredy_img_dft)
ans = -filteredy_img_back.real - filteredx_img_back.real
plt.subplot(111), plt.imshow(ans, cmap='gray')
plt.title("Final image"), plt.xticks([]), plt.yticks([])
plt.show()











# height, width = img.shape[:2]
# log_sobel_filter_dft = np.log(np.abs(sobel_filter_dft) + 1)
# sobel_dft = cv2.resize(log_sobel_filter_dft, (height, width), interpolation=cv2.INTER_LINEAR)

# plot_dft(sobel_dft, "Sobel Filter")

# cv2.imshow("Original", img)
# img = np.array(img, dtype=np.int16)

# hg_img = apply_horizontal_gradient(img)
# hg_img = hg_img.astype(np.uint8, copy=False)
# # cv2.imshow("After applying horizontal gradient", hg_img)
#
# vg_img = apply_vertical_gradient(img)
# vg_img = vg_img.astype(np.uint8, copy=False)
# # cv2.imshow("After applying vertical gradient", vg_img)