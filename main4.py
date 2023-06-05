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
    # hu_moments = cv2.HuMoments(moments).flatten()
    # mean = np.mean(patch_flat, axis=0)
    # std = np.std(patch_flat, axis=0)
    # moments = cv2.moments(patch_flat)
    hu_moments = cv2.HuMoments(moments).flatten()

    return [variance, *moments, *image_moments, *hu_moments]


def train():
    study_set = ['01_h', '02_h', '03_h']#, '04_h', '05_h', '06_h', '07_h']  # , '08_h', '09_h', '10_h']

    datasets = []
    mask_pixels = []
    for i, image_path in enumerate(study_set):
        dataset = []
        image = cv2.imread(f'images/{image_path}.jpg')
        mask = cv2.imread('healthy_fovmask/01_h_mask.tif')
        expected_result_mask = cv2.imread(f'healthy_manualsegm/{image_path}.tif')
        desired_width = 2000
        desired_height = 1800

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

    X_train, X_test, y_train, y_test = train_test_split(datasets, mask_pixels, test_size=0.2)
    print('datasets:', len(datasets), 'x', len(datasets[0]))
    print('mask_pixels:', len(mask_pixels))

    print("Standardizing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    force_training = True
    try:
        if force_training:
            raise FileNotFoundError
        # classifier = load_classifier("classifier.pkl")
        classifier = joblib.load('classifier.pkl')
        print("Model loaded from file...")
    except FileNotFoundError:
        print("Training...")
        classifier = RandomForestClassifier(n_estimators=150, n_jobs=-1)  # 0.793
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

    image_2 = cv2.imread(f'images/15_h.jpg')
    mask = cv2.imread('healthy_fovmask/15_h_mask.tif')
    expected_result_mask = cv2.imread(f'healthy_manualsegm/15_h.tif')
    desired_width = 1800
    desired_height = 1500

    # Resize all images
    # image_2 = cv2.resize(copy.deepcopy(image), (desired_width, desired_height))
    mask = cv2.resize(mask, (desired_width, desired_height))
    expected_result_mask = cv2.resize(expected_result_mask, (desired_width, desired_height))

    green_channel_image = image_2[:, :, 1]
    expected_result = expected_result_mask[:, :, 1]
    image_contrasted = cv2.equalizeHist(green_channel_image)

    #
    # image_contrasted[green_channel_mask == 0] = 0
    #

    # 5x5 patches of image
    patches = extract_patches(image_contrasted)

    # Middle pixel from mask patches
    result_patches_pixels = extract_expect_central_pixel(expected_result)

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
    pil_image = Image.fromarray(normalized_image)
    pil_image = pil_image.convert('L')
    pil_image.save("result.png")
    pil_image.show()



if __name__ == '__main__':
    main()
