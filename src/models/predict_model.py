"""
This script contains code for making predictions using a pre-trained machine learning model.
It includes functions to load model parameters from a YAML file and prepare data for model input.
"""

from pathlib import Path

import mlflow

import tensorflow as tf
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, log_loss

from utils import read_data_preparation_params, load_dataset, TEST_DATA_DIR

#from codecarbon import EmissionsTracker

model_folder_path = Path("models")

mlflow.set_experiment("cifar10")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

with mlflow.start_run():
    mlflow.log_params({
        "model_name": "NN_ImageNet_base_cifar10",
        "dataset_name": "CIFAR-10 Test"  
    })
    # Path of the test data folder
    input_folder_path = TEST_DATA_DIR

    # Get preparation parameters
    params = read_data_preparation_params("train")

    # `random_state` for the sake of reproducibility for tensorflow and numpy.
    tf.random.set_seed(params["random_state"])
    np.random.seed(params["random_state"])

    # Load the model
    loaded_model = tf.keras.models.load_model(model_folder_path / "imagenet_model_cifar10.h5")

    # Specify the model
    if params["algorithm"] == "InceptionV3":
        algorithm = tf.keras.applications.InceptionV3
        input_size = (params["input_size"], params["input_size"])
        input_shape = (params["input_size"], params["input_size"], 3)

    # Get Test dataset as Generator
    test_generator = load_dataset(params, input_folder_path, input_size)

    evaluation = loaded_model.evaluate(test_generator)

    loss = evaluation[0]
    accuracy = evaluation[1]
    mlflow.log_metric("evaluation_loss", loss)
    mlflow.log_metric("evaluation_accuracy", accuracy)

    # Calculate F1 score
    y_true = test_generator.classes
    y_pred = loaded_model.predict(test_generator)
    y_pred = y_pred.argmax(axis=-1)  # Convert predicted probabilities to class labels
    f1score = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    mlflow.log_metric("f1_score", f1score)
    mlflow.log_metric("accuracy_score", accuracy)
