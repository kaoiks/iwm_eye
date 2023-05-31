import cv2
import copy
import matplotlib
import numpy as np
from skimage import io, filters
from matplotlib import pyplot as plt

def main():
    #image = io.imread('images/oko2.png')
    image = cv2.imread('images/01_h.jpg')  # Read the image in grayscale
    mask = cv2.imread('healthy_fovmask/01_h_mask.tif')

    # image = io.imread('images/11_h.jpg')
    green_channel_image = image[:, :, 1]  # Select the green channel (index 1)
    green_channel_mask = mask[:, :, 1]

    image_copy = copy.deepcopy(green_channel_image)

    masked_image = cv2.bitwise_and(image_copy, image_copy, mask=green_channel_mask)



    blurred_image = cv2.GaussianBlur(masked_image, (5, 5), 2)
    edges = cv2.Canny(blurred_image, 5, 18)


    negative_image = 255 - edges

    # # Display the negative image
    plt.imshow(negative_image, cmap='gray')
    plt.title('Negative Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
