import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from random import sample

from skimage import filters
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
        temp_for = []
        for y in range(int(np.sqrt(len(X)))):
            temp_for.append(float(predictions[y + int(np.sqrt(len(X))) * x]))
        outcome.append(temp_for)
    outcome = np.asarray(outcome)
    # outcome = cv2.resize(outcome, image.shape[:2], interpolation=cv2.INTER_LINEAR)
    return outcome


def confusion_matrix(original_image, predicted_image, expected_image):
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    purple = (255, 0, 255)
    conf_matrix = np.zeros(original_image.shape, dtype=np.uint8)

    predicted_mask = (predicted_image != 0)
    model_mask = (expected_image != 0)

    conf_matrix[predicted_mask & model_mask] = green
    conf_matrix[predicted_mask & ~model_mask] = red
    conf_matrix[~predicted_mask & model_mask] = purple
    conf_matrix[~predicted_mask & ~model_mask] = white

    return conf_matrix


def image_metrics(predicted, model):
    assert predicted.shape == model.shape, "Predicted and model arrays must have the same shape."

    true_positive = np.sum((predicted > 0) & (model > 0))
    false_positive = np.sum((predicted > 0) & (model == 0))
    false_negative = np.sum((predicted == 0) & (model > 0))
    true_negative = np.sum((predicted == 0) & (model == 0))
    print(false_positive, true_positive)
    total = true_positive + false_positive + false_negative + true_negative
    accuracy = round((true_positive + true_negative) / total, 4)
    sensitivity = round(true_positive / (true_positive + false_negative + 1), 4)
    specificity = round(true_negative / (false_positive + true_negative + 1), 4)
    precision = round(true_positive / (true_positive + false_positive + 1), 4)
    g_mean = round(np.sqrt(sensitivity * specificity), 4)
    f_measure = round((2 * precision * sensitivity) / (precision + sensitivity + 1), 4)

    # Printing metrics
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"G-mean: {g_mean}")
    print(f"F-measure: {f_measure}")


def main():
    # Wymagania 3.0
    image = cv2.imread('images/12_h.jpg')
    mask = cv2.imread('healthy_fovmask/12_h_mask.tif')
    expected_result = cv2.imread('healthy_manualsegm/12_h.tif')
    desired_width = 1200
    desired_height = 800

    #Resize all images
    image_2 = cv2.resize(copy.deepcopy(image), (desired_width, desired_height))
    mask = cv2.resize(mask, (desired_width, desired_height))
    expected_result = cv2.resize(expected_result, (desired_width, desired_height))

    #Green channel
    expected_result = expected_result[:, :, 1]
    green_channel_image = image_2[:, :, 1]
    green_channel_mask = mask[:, :, 1]

    green_channel_image[green_channel_mask == 0] = 0

    image_copy = copy.deepcopy(green_channel_image)

    #Image processing
    blurred_image = filters.unsharp_mask(image_copy)
    sato = filters.sato(blurred_image)
    sato[green_channel_mask == 0] = 0
    threshold_value = 0.014
    predicted_image = ((sato > threshold_value).astype(int) * 255)

    plt.imshow(predicted_image, cmap='gray')
    plt.show()
    differences = confusion_matrix(image_2, predicted_image, expected_result)
    plt.imshow(differences)
    plt.show()

    image_metrics(predicted_image, expected_result)


if __name__ == '__main__':
    main()
