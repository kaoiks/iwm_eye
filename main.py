import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def main():

    image = cv2.imread('images/01_h.jpg')
    mask = cv2.imread('healthy_fovmask/01_h_mask.tif')

    green_channel_image = image[:, :, 1]
    green_channel_mask = mask[:, :, 1]

    image_copy = copy.deepcopy(green_channel_image)

    masked_image = cv2.bitwise_and(image_copy, image_copy, mask=green_channel_mask)

    blurred_image = cv2.GaussianBlur(masked_image, (5, 5), 2)
    edges = cv2.Canny(blurred_image, 5, 18)

    negative_image = 255 - edges

    plt.imshow(negative_image, cmap='gray')
    plt.title('Negative Image')
    plt.axis('off')
    plt.show()

    # Wymagania 4.0

    size = 5
    image = blurred_image
    if len(image.shape) == 2:  # Grayscale image
        height, width = image.shape
    else:  # Color image
        height, width, _ = image.shape
    sliced_parts = []
    for y in range(0, height, size):
        for x in range(0, width, size):
            if len(image.shape) == 2:  # Grayscale image
                part = image[y:y + size, x:x + size]
            else:  # Color image
                part = image[y:y + size, x:x + size, :]
            sliced_parts.append(part)

    parameters = []
    # masked_image was here
    mean_value = np.mean(blurred_image)

    max_value = np.max(blurred_image)

    min_value = np.min(blurred_image)

    parameters = [mean_value, max_value, min_value]

    # Hu Moments
    _, part = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else None

    if contour is not None:
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        parameters.extend(hu_moments.flatten())





    Model = KNeighborsClassifier(n_neighbors=3)




    # # load images to an array
    # images = []
    # for i in range(1, 10):
    #     img = cv2.imread('images/0' + str(i) + '_h.jpg')
    #     images.append(img)
    #
    # # load masks to an array
    # masks = []
    # for i in range(1, 10):
    #     mask = cv2.imread('healthy_fovmask/0' + str(i) + '_h_mask.tif')
    #     masks.append(mask)
    #
    # # convert all images and masks to green channel
    # green_channel_images = []
    # green_channel_masks = []
    # for i in range(0, 9):
    #     green_channel_images.append(images[i][:, :, 1])
    #     green_channel_masks.append(masks[i][:, :, 1])
    #
    # # apply masks to images
    # masked_images = []
    # for i in range(0, 9):
    #     masked_images.append(cv2.bitwise_and(green_channel_images[i], green_channel_images[i], mask=green_channel_masks[i]))
    #
    #
    # # apply gaussian blur to masked_images
    # blurred_images = []
    # for i in range(0, 9):
    #     blurred_images.append(cv2.GaussianBlur(masked_images[i], (5, 5), 2))

if __name__ == '__main__':
    main()
