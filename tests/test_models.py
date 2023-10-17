import pickle,pytest

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

from src import MODELS_DIR, TEST_DATA_DIR
from src.models.utils import load_dataset, readDataPreparationParams


@pytest.fixture
def cifar_model_pickle():
    with open(MODELS_DIR / "imagenet_model_cifar10.pkl", "rb") as f:
        return pickle.load(f)
    
@pytest.fixture
def cifar_model_h5():
    return load_model(MODELS_DIR / "imagenet_model_cifar10.h5")

@pytest.fixture
def cifar10_validation_data():
    params = readDataPreparationParams("train")
    return load_dataset(params, TEST_DATA_DIR, (32, 32))


def test_cifar_model(cifar_model_h5, cifar10_validation_data):
    y_pred = cifar_model_h5.predict(cifar10_validation_data)
    y_pred = y_pred.argmax(axis=-1)  # Convert predicted probabilities to class labels
    f1score = f1_score(y_true, y_pred, average='weighted')
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    assert f1score >= 0.1
    assert mae < 3.3
    assert mse < 16.5
    
def test_classification_cifar_model(cifar_model_h5, cifar10_validation_data):
    y_pred_single = cifar_model_h5.predict(cifar10_validation_data)
    assert y_pred_single[0].argmax() == cifar10_validation_data.classes[0], "Prediction does not match the correct label."