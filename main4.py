import pickle
import time

import cv2
import copy

import joblib
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from PIL import Image
from skimage import filters
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy import stats
import cv2
from sklearn.preprocessing import StandardScaler


def confusion_matrix(original_image, predicted_image, expected_image):
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    purple = (255, 0, 255)
    conf_matrix = np.zeros(predicted_image.shape)

    predicted_mask = (predicted_image != 0)
    model_mask = (expected_image != 0)

    conf_matrix[predicted_mask & model_mask] = green
    conf_matrix[predicted_mask & ~model_mask] = red
    conf_matrix[~predicted_mask & model_mask] = purple
    conf_matrix[~predicted_mask & ~model_mask] = white

    return conf_matrix


def confusion_matrix2(original_image, predicted_image, expected_image):
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    purple = (255, 0, 255)
    conf_matrix = np.array(predicted_image.shape[:2])
    result_array = []
    for i in range(predicted_image.shape[0]):
        result_row = []
        for j in range(predicted_image.shape[1]):
            if predicted_image[i, j] and expected_image[i, j]:
                result_row.append(green)
                # conf_matrix[i, j] = green  # Green
            elif predicted_image[i, j] and not expected_image[i, j]:
                result_row.append(red)
                # conf_matrix[i, j] = red  # Red
            elif not predicted_image[i, j] and expected_image[i, j]:
                result_row.append(purple)
                # conf_matrix[i, j] = purple  # Purple
            else:
                result_row.append(white)
                # conf_matrix[i, j] = white  # White
        result_array.append(result_row)

    return np.array(result_array)


def extract_patches(image, patch_size=5):
    # Splitting the image into patches

    rows, cols = image.shape[0], image.shape[1]

    patches = []
    for i in range(0, rows - patch_size + 1, patch_size):
        for j in range(0, cols - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches


def extract_expect_central_pixel(image, patch_size=5):
    # Splitting the image into patches and getting the central pixel

    rows, cols = image.shape[0], image.shape[1]
    half = patch_size // 2

    pixels = []
    for i in range(half, rows - half, patch_size):
        for j in range(half, cols - half, patch_size):
            pixel = int(image[i, j])
            pixels.append(pixel)

    return pixels


def extract_features(patch_flat):
    variance = np.var(patch_flat)
    moments = stats.moment(patch_flat, moment=[1, 2, 3])
    image_moments = list(cv2.moments(patch_flat).values())
    hu_moments = cv2.HuMoments(moments).flatten()

    return [variance, *moments, *image_moments, *hu_moments]


def train():
    study_set = ['01_h', '02_h', '03_h']#, '04_h']#, '05_h', '06_h', '07_h', '08_h', '09_h', '10_h']

    datasets = []
    mask_pixels = []
    for i, image_path in enumerate(study_set):
        dataset = []
        image = cv2.imread(f'images/{image_path}.jpg')
        mask = cv2.imread('healthy_fovmask/01_h_mask.tif')
        expected_result_mask = cv2.imread(f'healthy_manualsegm/{image_path}.tif')
        desired_width = 3500
        desired_height = 2300

        # Resize all images
        image_2 = cv2.resize(copy.deepcopy(image), (desired_width, desired_height))
        mask = cv2.resize(mask, (desired_width, desired_height))
        expected_result_mask = cv2.resize(expected_result_mask, (desired_width, desired_height))

        green_channel_image = image_2[:, :, 1]
        expected_result = expected_result_mask[:, :, 1]
        image_contrasted = cv2.equalizeHist(green_channel_image)

        # 5x5 patches of image
        patches = extract_patches(image_contrasted)

        # Middle pixel from mask patches
        result_patches_pixels = extract_expect_central_pixel(expected_result)

        for patch in patches:
            flat_patch = patch.flatten().astype(float)
            features = extract_features(flat_patch)
            dataset.append(list(flat_patch) + features)

        datasets.extend(dataset)
        mask_pixels.extend(result_patches_pixels)

    X_train, X_test, y_train, y_test = train_test_split(datasets, mask_pixels, test_size=0.3)
    print('datasets:', len(datasets), 'x', len(datasets[0]))
    print('mask_pixels:', len(mask_pixels))

    print("Standardizing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = RandomForestClassifier(n_estimators=250, n_jobs=-1)
    classifier.fit(X_train_scaled, y_train)

    # Klasyfikacja danych testowych
    print("Predicting...")
    y_pred = classifier.predict(X_test_scaled)

    # Obliczenie dokładności klasyfikacji
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    joblib.dump(scaler, 'scaler.bin', compress=3)
    # with open("classifier.pkl", 'wb') as file:
    #     pickle.dump(classifier, file)
    joblib.dump(classifier, 'classifier.pkl', compress=3)


def main():
    # train()

    classifier = joblib.load('classifier.pkl')
    scaler = joblib.load('scaler.bin')

    image_2 = cv2.imread(f'images/12_h.jpg')
    mask = cv2.imread('healthy_fovmask/12_h_mask.tif')
    expected_result_mask = cv2.imread(f'healthy_manualsegm/12_h.tif')

    image_2 = cv2.resize(copy.deepcopy(image_2), (2000, 1500))

    height, width, channels = image_2.shape
    desired_width = width
    desired_height = height

    # Resize all images

    # mask = cv2.resize(mask, (desired_width // 5, desired_height // 5))
    # expected_result_mask = cv2.resize(expected_result_mask, (desired_height, desired_width))

    green_channel_image = image_2[:, :, 1]
    green_channel_mask = mask[:, :, 1]
    expected_result_mask = expected_result_mask[:, :, 1]
    # expected_result = expected_result_mask[:, :, 1]

    expected_result_mask = cv2.resize(expected_result_mask, (desired_width // 5, desired_height // 5))

    image_contrasted = cv2.equalizeHist(green_channel_image)

    #
    # image_contrasted[green_channel_mask == 0] = 0
    #

    # 5x5 patches of image
    patches = extract_patches(image_contrasted)

    # Middle pixel from mask patches
    result_patches_pixels = extract_expect_central_pixel(expected_result_mask)

    dataset = []
    for patch in patches:
        flat_patch = patch.flatten().astype(float)
        features = extract_features(flat_patch)
        dataset.append(list(flat_patch) + features)

    X_test_scaled = scaler.transform(dataset)

    print("Predicting...")
    y_pred = classifier.predict(X_test_scaled)
    print("Predicted...")
    height, width, channels = image_2.shape
    new_image = np.zeros((height // 5, width //5))
    for x in range(height // 5):
        for y in range(width // 5):
            new_image[x, y] = y_pred[0]
            y_pred = y_pred[1:]

    normalized_image = (new_image / np.max(new_image)) * 255
    # normalized_image[green_channel_mask == 0] = 0

    pil_image = Image.fromarray(normalized_image)
    pil_image = pil_image.convert('L')
    pil_image.save("result.png")

    differences = confusion_matrix2(image_2, normalized_image, expected_result_mask)
    plt.imshow(differences)
    plt.show()

    pil_image.show()


if __name__ == '__main__':
    main()
