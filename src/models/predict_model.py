from pathlib import Path

import mlflow, time, pickle, sys, yaml

import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import InceptionV3, VGG16
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score  # Import f1_score


#from codecarbon import EmissionsTracker

sys.path.append('/home/w4z3/Git/MLOPS-cifar10')
from src import MODELS_DIR, TEST_DATA_DIR
from utils import *

model_folder_path = Path("models")

mlflow.set_experiment("cifar10")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

with mlflow.start_run():
    # Path of the test data folder
    input_folder_path = TEST_DATA_DIR

    # Get preparation parameters
    params = readDataPreparationParams("train")

    # Load the model
    loaded_model = load_model(model_folder_path / "imagenet_model_cifar10.h5")
    
    # Specify the model
    if params["algorithm"] == "InceptionV3":
        algorithm = InceptionV3
        input_size = (75, 75)
        input_shape = (75, 75, 3)
    elif params["algorithm"] == "VGG16":
        algorithm = VGG16
        input_size = (32, 32)
        input_shape = (32, 32, 3)

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
    mlflow.log_metric("evaluation_f1score", f1score)
