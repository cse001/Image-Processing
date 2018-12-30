import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as read
import cv2

MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255


def load_image(name, usecv=True, type=None):
    if usecv:
        if type is not None:
            return cv2.imread(name, type)
        else:
            return cv2.imread(name)
    else:
        return read.imread(name)


def plot_image(image, usecv=True):
    if usecv:
        cv2.imshow('image', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.show()


def plot_figures(figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.
    Parameters
    ----------
    figures : list of (name, image)
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, figure in enumerate(figures):
        name, image = figure
        axeslist.ravel()[ind].imshow(image, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(name)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional
    plt.show()


def mean_filter(arr):
    return np.clip(np.rint(np.mean(arr)), 0, 255)


def median_filter(arr):
    return np.rint(np.median(arr))


def max_filter(arr):
    return np.max(arr)


def min_filter(arr):
    return np.min(arr)


def apply_filter(img, filter_size, function):
    filter_size = filter_size // 2

    rows, cols = img.shape

    img_new = img.copy()

    for r in range(filter_size, rows - filter_size):
        for c in range(filter_size, cols - filter_size):
            img_new[r, c] = function(img[r - filter_size: r + filter_size + 1, c - filter_size: c + filter_size + 1])

    return img_new.astype(np.uint8)


def mean_square_error(img1, img2):
    if img1.shape != img2.shape:
        print("Cannot calculate mean square error as both images do not have same shape")
        return

    rows, cols = img2.shape
    error = 0

    err_img = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            err_img[r, c] = (img1[r, c].astype(np.float64) - img2[r, c].astype(np.float64)) ** 2
            error += err_img[r, c]

    error /= rows * cols

    max_err = np.max(err_img)
    min_err = np.min(err_img)

    err_img = ((err_img - min_err) / (max_err - min_err)) * MAX_PIXEL_VALUE

    return err_img.astype(np.uint8), error


def linear_transformation_to_pixel_value_range(data):
    max = np.max(data)
    min = np.min(data)
    data = np.array(data, dtype=np.float64)
    new_data = MAX_PIXEL_VALUE * ((data - min) / (max - min))
    return np.rint(new_data).astype(np.uint8)


def remove_noise(img, img_name, filter_size, filter_function):
    img_filtered = apply_filter(img.copy(), filter_size, filter_function)

    img_err, err_left_mean = mean_square_error(img.copy(), img_filtered.copy())

    img_diff = img_filtered.astype(np.int) - img.astype(np.int)
    img_diff = linear_transformation_to_pixel_value_range(img_diff)

    figures = []

    figures.append((img_name + ' orginal image', img))
    figures.append((str(filter_function)[9:-19] + 'ed', img_filtered))
    figures.append(('Error image', img_diff))

    # print('MSE of ' + img_name + ' image after applying ' + str(filter_function)[9:-19] + ' = ' + str(err_left_mean))

    plot_figures(figures, 1, 3)


img = load_image('lab 3/img1noisy.tif', type=0)

rows, cols = img.shape
centre_col = cols // 2

img_left = img[:, : centre_col + 1]
img_right = img[:, centre_col:]

remove_noise(img_left, 'Left', 3, max_filter)
remove_noise(img_left, 'Left', 3, min_filter)
remove_noise(img_left, 'Left', 3, mean_filter)
remove_noise(img_left, 'Left', 3, median_filter)

remove_noise(img_right, 'Right', 3, max_filter)
remove_noise(img_right, 'Right', 3, min_filter)
remove_noise(img_right, 'Right', 3, mean_filter)
remove_noise(img_right, 'Right', 3, median_filter)
