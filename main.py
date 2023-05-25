import cv2
import copy
import matplotlib
import numpy as np
from skimage import filters
from matplotlib import pyplot as plt

def main():
    image = cv2.imread('images/oko2.png')
    # image = cv2.imread('images/11_h.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred_image = cv2.GaussianBlur(image, (5, 5), 2)
    edges = cv2.Canny(blurred_image, 5, 18)
    negative_image = 255 - edges

    # Display the negative image
    plt.imshow(negative_image, cmap='gray')
    plt.title('Negative Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

