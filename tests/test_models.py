"""
This module contains test fixtures and functions for evaluating a CIFAR-10 image 
classification model.

It includes fixtures for loading pre-trained models and validation datasets, 
as well as test functions that assess the model's performance using metrics 
like F1 score, mean absolute error, and mean squared error. These tests are designed 
to ensure the model's effectiveness in classifying CIFAR-10 images.

Fixtures:
- `cifar_model_pickle`: Loads a pre-trained model from a pickle file.
- `cifar_model_h5`: Loads a pre-trained model from an HDF5 file.
- `cifar10_validation_data`: Loads CIFAR-10 validation data for testing the model.

Test Functions:
- `test_cifar_model`: Evaluates the CIFAR-10 model's performance, calculating and asserting 
    F1 score, mean absolute error, and mean squared error.
- `test_classification_cifar_model`: Tests the classification of a single image and asserts 
    that the prediction matches the correct label.

This module is designed for testing and evaluating the model's capabilities in image 
classification tasks, particularly for CIFAR-10 dataset.
"""

import numpy as np
import pytest
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image

from sklearn.metrics import f1_score, accuracy_score

from src.app.classes import ImagesType
from src.models.utils import load_dataset, read_data_preparation_params, get_model
from src import MODELS_DIR, TEST_DATA_DIR

# Constants
F1_SCORE_THRESHOLD = 0.6
ACCURACY_THRESHOLD = 0.7

@pytest.fixture
def cifar_model_h5():
    """
    Fixture to load a pre-trained CIFAR-10 model from an HDF5 file.

    Returns:
        object: The pre-trained model loaded from the HDF5 file.
    """
    return get_model()

@pytest.fixture
def cifar10_validation_data():
    """
    Fixture to load CIFAR-10 validation data for testing the model.

    Returns:
        object: A dataset generator containing CIFAR-10 validation data.
    """
    params = read_data_preparation_params("train")
    return load_dataset(params, TEST_DATA_DIR, (32, 32))


def test_classification_cifar_model(cifar_model_h5):
    """
    Test function to validate the classification of a single CIFAR-10 image and ensure that 
    the prediction matches the correct label.

    Args:
        cifar_model_h5 (object): The pre-trained CIFAR-10 model.
        cifar10_validation_data (object): CIFAR-10 validation data.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "../data/test/airplane/0001.jpg")
    
    img = image.img_to_array(Image.open(image_path).resize((75, 75)))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    prediction_probabilities = cifar_model_h5.predict(img_array)[0]

    predicted_class = ImagesType(np.argmax(prediction_probabilities)).name

    assert predicted_class == "AIRPLANE", "Prediction does not match the correct label."
