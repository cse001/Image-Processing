import cv2
import matplotlib.pyplot as plt

# Write a function, generateHistogram, which generates the histogram of an image. The
# function should take an image data array (with pixel values in the range 0 – 255) as its only
# parameter and return an array containing the histogram of the image. The histogram can be
# displayed using the built in plotting function. Use this new function to generate and display
# histograms for the following images.


def generate_histogram(orig):
    hist = [0] * 256
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            hist[orig[i][j]] = hist[orig[i][j]] + 1
    return hist

img1 = cv2.imread('lab2_images/top_left.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('lab2_images/bottom_left.tif', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('lab2_images/2nd_from_top.tif', cv2.IMREAD_GRAYSCALE)

fig = plt.figure(figsize=(20, 12))
generated_hist1 = generate_histogram(img1)
x = range(len(generated_hist1))
fig.add_subplot(2, 2, 1)
plt.bar(x, generated_hist1, 1, color="blue")
generated_hist2 = generate_histogram(img2)
fig.add_subplot(2, 2, 2)
plt.bar(x, generated_hist2, 1 / 1.5, color="red")
generated_hist3 = generate_histogram(img3)
fig.add_subplot(2, 2, 3)
plt.bar(x, generated_hist3, 1 / 1.5, color="green")
plt.show()
