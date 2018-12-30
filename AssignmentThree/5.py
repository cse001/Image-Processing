import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lab 3/img2.tif', 0)
img_original = cv2.imread('lab 3/img1original.tif', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
img_dft = 20 * np.log(np.abs(fshift))

img_diff = img - img_original

rows, cols = img.shape
centre_row, centre_col = rows // 2, cols // 2

y1 = centre_row + 16
y2 = centre_row - 16
mask2 = np.ones((rows, cols), np.uint8)
mask2[centre_row, y1] = 0
mask2[centre_row, y2] = 0
ifshift2 = fshift * mask2
f_ishift2 = np.fft.ifftshift(ifshift2)
mag_back2 = np.fft.ifft2(f_ishift2)
img_back2 = np.abs(mag_back2)

plt.subplot(1, 3, 1), plt.imshow(img_back2, cmap='gray')
plt.title('Output Image '), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(img_dft, cmap='gray')
plt.title('FFT'), plt.xticks([]), plt.yticks([])

plt.show()

f = np.fft.fft2(img_diff)
fshift = np.fft.fftshift(f)
img_diff_dft = 20 * np.log(np.abs(fshift))
plt.subplot(1, 1, 1), plt.imshow(img_diff_dft, cmap='gray')
plt.title('Difference'), plt.xticks([]), plt.yticks([])

plt.show()
