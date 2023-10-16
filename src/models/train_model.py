from pathlib import Path

import mlflow, time, pickle, sys, yaml

import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import InceptionV3, VGG16

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

#from codecarbon import EmissionsTracker

sys.path.append('/home/w4z3/Git/MLOPS-cifar10')
from src import MODELS_DIR, TRAIN_DATA_DIR
from utils import *

mlflow.set_experiment("cifar10")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

with mlflow.start_run():
    # Path of the prepared data folder
    input_folder_path = TRAIN_DATA_DIR

    # Get preparation parameters
    params = readDataPreparationParams("train")

    # `random_state` for the sake of reproducibility for tensorflow and numpy.
    tf.random.set_seed(params["random_state"])
    np.random.seed(params["random_state"])

    # ============== #
    # MODEL TRAINING #
    # ============== #

    # Specify the model
    if params["algorithm"] == "InceptionV3":
        algorithm = InceptionV3
        input_size = (75, 75)
        input_shape = (75, 75, 3)
    elif params["algorithm"] == "VGG16":
        algorithm = VGG16
        input_size = (32, 32)
        input_shape = (32, 32, 3)

    # Get Train dataset as Generator
    train_generator = load_dataset(params, input_folder_path, input_size)

    imagenet_base_model = algorithm(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in imagenet_base_model.layers:
        layer.trainable = False

    x = Flatten()(imagenet_base_model.output)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    imagenet_model = Model(inputs=imagenet_base_model.input, outputs=predictions)
    imagenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
        
    imagenet_model.fit(train_generator, epochs=params["num_epochs"], batch_size=params["batch_size"])

    # Save the model as a pickle file
    Path("models").mkdir(exist_ok=True)
    with open(MODELS_DIR / "imagenet_model_cifar10.pkl", "wb") as pickle_file:
        pickle.dump(imagenet_model, pickle_file)

    # Save the model with TensorFlow methods to ensure compatibility and integrity. 
    imagenet_model.save(MODELS_DIR/'imagenet_model_cifar10.h5')