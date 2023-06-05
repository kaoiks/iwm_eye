import cv2
import copy
import matplotlib
import numpy as np
from skimage import filters
import skimage
from matplotlib import pyplot as plt
# wyrzucic obramowaine maska field of view
# sprobowac wypelnic linie
def main():
    image = skimage.io.imread('images/oko2.png')
    image = image[:, :, 1]

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 2)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 5, 25)

    # Invert the image to get negative
    negative_image = 255 - edges


    # Display the negative image with filled lines
    plt.imshow(negative_image, cmap='gray')
    plt.title('Filled Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

