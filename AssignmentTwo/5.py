import cv2
import numpy as np

# Try to enhance the ‘skeleton.jpg’ image by performing sequence of operations as
# discussed in the textbook/class or any other alternative method.


def average_filter(orig, size):
    arr = np.zeros(orig.shape, dtype=np.uint8)
    avg_fil = np.ones([size, size])
    avg_fil = avg_fil / (size ** 2)
    rows = orig.shape[0]
    cols = orig.shape[1]
    pad = size // 2
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            arr[i][j] = np.sum(np.multiply(orig[i - pad: i + pad + 1, j - pad: j + pad + 1], avg_fil))

    return arr


def laplacian(img):
    laplacian_image = cv2.Laplacian(img, cv2.BORDER_CONSTANT)
    cv2.imshow('Laplacian', laplacian_image)
    return laplacian_image


def sobel(img):
    sobelx = cv2.Sobel(img, cv2.BORDER_CONSTANT, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.BORDER_CONSTANT, 0, 1, ksize=3)
    sobel = cv2.add(sobelx, sobely)
    cv2.imshow('Sobel', sobel)
    return sobel


def power_law_transformation(img):
    c = 1
    g = 0.4
    h, w = img.shape[:2]
    img = img / 255.0
    for i in range(h):
        for j in range(w):
            p = img[i, j]
            s = c * (p ** g)
            img[i, j] = s
    cv2.imshow('Power Law with gamma : ' + str(g), img)
    return img


img = cv2.imread("lab2_images/skeleton.tif", 0)
cv2.imshow('Original Image', img)
a = img
b = laplacian(a)
c = cv2.subtract(a, b)
cv2.imshow('Sharpened Image', c)
d = sobel(a)
e = average_filter(d, 5)
f = (c*e)//255
g = cv2.add(a, f)
cv2.imshow('Before Power Law Transformation', g)
h = power_law_transformation(g)
cv2.imshow('Final', h)
cv2.waitKey(0)
cv2.destroyAllWindows()
