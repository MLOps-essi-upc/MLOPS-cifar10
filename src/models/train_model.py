"""
This module contains the code for training a machine learning model.
It includes calls to functions for load model parameters from yaml file and
functions to load a generator for train and test the model.
"""
import pickle
from pathlib import Path

import mlflow

import tensorflow as tf
import numpy as np

from utils import read_data_preparation_params, load_dataset, MODELS_DIR, TRAIN_DATA_DIR
#from codecarbon import EmissionsTracker

mlflow.set_experiment("cifar10")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

with mlflow.start_run():
    mlflow.log_params({
        "model_name": "NN_ImageNet_base_cifar10",
        "dataset_name": "CIFAR-10 Train"  
    })

    # Path of the prepared data folder
    input_folder_path = TRAIN_DATA_DIR

    # Get preparation parameters
    params = read_data_preparation_params("train")

    # `random_state` for the sake of reproducibility for tensorflow and numpy.
    tf.random.set_seed(params["random_state"])
    np.random.seed(params["random_state"])

    # ============== #
    # MODEL TRAINING #
    # ============== #

    # Specify the model
    if params["algorithm"] == "InceptionV3":
        algorithm = tf.keras.applications.InceptionV3
        input_size = (params["input_size"], params["input_size"])
        input_shape = (params["input_size"], params["input_size"], 3)

    # Get Train dataset as Generator
    train_generator = load_dataset(params, input_folder_path, input_size)

    imagenet_base_model = algorithm(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in imagenet_base_model.layers:
        layer.trainable = False

    x = tf.keras.layers.Flatten()(imagenet_base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

    imagenet_model = tf.keras.Model(inputs=imagenet_base_model.input, outputs=predictions)
    imagenet_model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    imagenet_model.fit(train_generator,
                       epochs=params["num_epochs"],
                       batch_size=params["batch_size"])

    # Save the model as a pickle file
    Path("models").mkdir(exist_ok=True)
    with open(MODELS_DIR / "imagenet_model_cifar10.pkl", "wb") as pickle_file:
        pickle.dump(imagenet_model, pickle_file)

    # Save the model with TensorFlow methods to ensure compatibility and integrity.
    imagenet_model.save(MODELS_DIR/'imagenet_model_cifar10.h5')

    # Save artifact
    mlflow.tensorflow.log_model(imagenet_model, "cifar10_NN_imageNet_model")
