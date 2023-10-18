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

import pickle
import pytest

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

from src.models.utils import load_dataset, read_data_preparation_params
from src import MODELS_DIR, TEST_DATA_DIR


@pytest.fixture
def cifar_model_pickle():
    """
    Fixture to load a pre-trained CIFAR-10 model from a pickle file.

    Returns:
        object: The pre-trained model loaded from the pickle file.
    """
    with open(MODELS_DIR / "imagenet_model_cifar10.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture
def cifar_model_h5():
    """
    Fixture to load a pre-trained CIFAR-10 model from an HDF5 file.

    Returns:
        object: The pre-trained model loaded from the HDF5 file.
    """
    return tf.keras.models.load_model(MODELS_DIR / "imagenet_model_cifar10.h5")

@pytest.fixture
def cifar10_validation_data():
    """
    Fixture to load CIFAR-10 validation data for testing the model.

    Returns:
        object: A dataset generator containing CIFAR-10 validation data.
    """
    params = read_data_preparation_params("train")
    return load_dataset(params, TEST_DATA_DIR, (32, 32))


def test_cifar_model(cifar_model_h5, cifar10_validation_data):
    """
    Test function to evaluate the CIFAR-10 model's performance using metrics like F1 score, 
    mean absolute error, and mean squared error.

    Args:
        cifar_model_h5 (object): The pre-trained CIFAR-10 model.
        cifar10_validation_data (object): CIFAR-10 validation data.
    """
    y_true = cifar10_validation_data.classes
    y_pred = cifar_model_h5.predict(cifar10_validation_data)
    y_pred = y_pred.argmax(axis=-1)  # Convert predicted probabilities to class labels
    f1score = f1_score(y_true, y_pred, average='weighted')
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    assert f1score >= 0.1
    assert mae < 3.3
    assert mse < 16.5

def test_classification_cifar_model(cifar_model_h5, cifar10_validation_data):
    """
    Test function to validate the classification of a single CIFAR-10 image and ensure that 
    the prediction matches the correct label.

    Args:
        cifar_model_h5 (object): The pre-trained CIFAR-10 model.
        cifar10_validation_data (object): CIFAR-10 validation data.
    """
    y_pred_single = cifar_model_h5.predict(cifar10_validation_data)
    correct_class = cifar10_validation_data.classes[0]
    predicted_class = y_pred_single[0].argmax()
    assert predicted_class == correct_class, "Prediction does not match the correct label."
