import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.neighbors import KNeighborsClassifier


def slice(image, size=3):
    height = image.shape[0]
    width = image.shape[1]
    sliced_parts = []
    for y in range(0, height, size):
        for x in range(0, width, size):
            part = image[y:y + size, x:x + size]
            sliced_parts.append(part)
    return sliced_parts


def features(image):
    parameters = []

    mean_value = np.mean(image)

    max_value = np.max(image)

    min_value = np.min(image)

    parameters = [mean_value, max_value, min_value]

    _, part = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    moment = cv2.moments(part, True)
    hu_moments = cv2.HuMoments(moment).flatten()
    parameters.extend(hu_moments)

    return parameters


def process_picture(model, image, size=3):
    parts = slice(image, size)
    X = np.array([features(x) for x in parts])
    predictions = model.predict(X)

    outcome = list()
    for x in range(int(np.sqrt(len(X)))):
        temp_for=[]
        for y in range(int(np.sqrt(len(X)))):
            temp_for.append(float(predictions[y+int(np.sqrt(len(X)))*x][0]))
        outcome.append(temp_for)
    outcome=np.asarray(outcome)
    outcome = cv2.resize(outcome, image.shape[:2], interpolation= cv2.INTER_LINEAR)
    return outcome


def main():

    # Wymagania 3.0
    image = cv2.imread('images/01_h.jpg')
    mask = cv2.imread('healthy_fovmask/01_h_mask.tif')

    green_channel_image = image[:, :, 1]
    green_channel_mask = mask[:, :, 1]

    image_copy = copy.deepcopy(green_channel_image)

    masked_image = cv2.bitwise_and(image_copy, image_copy, mask=green_channel_mask)

    blurred_image = cv2.GaussianBlur(masked_image, (5, 5), 2)
    edges = cv2.Canny(blurred_image, 5, 18)

    negative_image = 255 - edges

    # Wymagania 4.0
    binary_labels = cv2.imread('healthy_manualsegm/01_h.tif')

    a_image = np.array(image)
    a_binary_labels = np.array(binary_labels)
    size = 3

    a_image = cv2.resize(a_image, (512, 512))
    a_binary_labels = cv2.resize(a_binary_labels, (512, 512))

    a_image = cv2.cvtColor(a_image, cv2.COLOR_RGB2GRAY)

    parts = slice(a_image, size)
    result = slice(a_binary_labels, size)

    samples = sample(range(0, len(parts)), len(parts) // 12)

    parameters, decisions = list(), list()

    for x in samples:
        parameters.append(features(parts[x]))
        decisions.append(result[x][size // 2][size // 2])
    decisions = list(np.squeeze(np.asarray(decisions)))
    decisions = np.logical_and(decisions, decisions)

    X = parameters[:len(parameters) // 4]
    X_test = parameters[len(parameters) // 4:]

    Y = decisions[:len(parameters) // 4]
    Y_test = decisions[len(parameters) // 4:]

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(X, Y)

    prediction = model.predict(X_test)
    result1 = np.sum(prediction == Y_test) / len(prediction)

    print('-' * 20)
    print(f'Result of test: {round(result1, 9)*100}%')
    print('-' * 20)

    #image3 = cv2.imread('images/04_h.jpg')
    #a_image3 = np.array(image3)
    #a_image3 = cv2.resize(a_image3, (512, 512))
    #a_image3 = cv2.cvtColor(a_image3, cv2.COLOR_RGB2GRAY)

    knns = process_picture(model, a_image, size)

    knns = 255 - knns

    plt.subplot(1, 2, 1)
    plt.imshow(negative_image, cmap='gray')
    plt.title('Image filtered with Canny')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(knns, cmap='gray')
    plt.title('Image filtered with KNNs')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
