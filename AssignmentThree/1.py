import sys
import cv2
import numpy as np
import math


def gaussian_filter(x, y, centre_x, centre_y, radius):
    f = (x - centre_x) ** 2 + (y - centre_y) ** 2
    f = math.exp(-(f / (2 * (radius ** 2))))
    return f


img = cv2.imread('lab 3/97.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", img)

res = np.fft.fft2(img)
fshift = np.fft.fftshift(res)
img_fft = np.log(np.abs(fshift) + 1)

rows, cols = img.shape
centre_row_no, centre_col_no = int(rows / 2), int(cols / 2)
filter_type = sys.argv[1]    # L - low pass H - high pass
rad = float(sys.argv[2])     # R - radius(Dzero)
if filter_type == 'H':
    for i in range(rows):
        for j in range(cols):
            fshift[i, j] = fshift[i, j] * (1 - gaussian_filter(i, j, centre_row_no, centre_col_no, rad))
else:
    for i in range(rows):
        for j in range(cols):
            fshift[i, j] = fshift[i, j] * gaussian_filter(i, j, centre_row_no, centre_col_no, rad)


f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
if filter_type == 'H':
    cv2.imshow("After HPF", np.array(img + img_back, dtype=np.uint8))
else:
    cv2.imshow("After LPF", np.array(img_back, dtype=np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
