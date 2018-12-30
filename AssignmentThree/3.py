import sys
import cv2
import numpy as np

img = cv2.imread('lab 3/two_cats.jpg', cv2.IMREAD_GRAYSCALE)

noise_type = sys.argv[1]
if noise_type == 'G':
    row, col = img.shape
    mean = float(sys.argv[2])
    variance = float(sys.argv[3])
    sigma = variance ** 0.5
    gaussian = np.ndarray(img.shape, dtype=np.int16)
    cv2.randn(gaussian, mean, sigma)
    noisy_image = img + gaussian
    cv2.imshow("Original Image", np.array(img, dtype=np.uint8))
    cv2.imshow("Image after adding gaussian noise", np.array(noisy_image, dtype=np.uint8))

else:
    row, col = img.shape
    amount_of_noise = float(sys.argv[2])
    salt_noise_prob = float(sys.argv[3])
    output_image = np.copy(img)
    num_salt_noise = np.ceil(amount_of_noise * img.size * salt_noise_prob)
    xcoords = [np.random.randint(0, img.shape[0] - 1, int(num_salt_noise))]
    ycoords = [np.random.randint(0, img.shape[1] - 1, int(num_salt_noise))]
    for i in range(int(num_salt_noise)):
        for j in range(int(num_salt_noise)):
            output_image[xcoords, ycoords] = 255

    num_pepper_noise = np.ceil(amount_of_noise * img.size * (1. - salt_noise_prob))
    xcoords = [np.random.randint(0, img.shape[0] - 1, int(num_pepper_noise))]
    ycoords = [np.random.randint(0, img.shape[1] - 1, int(num_pepper_noise))]
    for i in range(int(num_pepper_noise)):
        for j in range(int(num_pepper_noise)):
            output_image[xcoords, ycoords] = 0
    cv2.imshow("Original", np.array(img, dtype=np.uint8))
    cv2.imshow("Salt and Pepper noise added", np.array(output_image, dtype=np.uint8))

cv2.waitKey(20000)
cv2.destroyAllWindows()
