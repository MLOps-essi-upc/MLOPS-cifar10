"""
This utility script provides functions for data preparation and model utility.
It includes functions for creating image generators, loading datasets, and
reading data preparation parameters from a YAML file.
"""

from pathlib import Path

import yaml

import tensorflow as tf


def create_image_generator(params):
    """
    Create an image data generator for data augmentation.

    Args:
        params (dict): A dictionary containing data augmentation parameters.

    Returns:
        tf.keras.preprocessing.image.ImageDataGenerator: An image data 
        generator configured with the specified parameters.
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
        zoom_range=params["zoom_range"],
        horizontal_flip=params["horizontal_flip"]
    )
    return datagen

def load_dataset(params, input_folder_path, input_size):
    """
    Load and configure a dataset generator for model training.

    Args:
        params (dict): A dictionary containing dataset and batch parameters.
        input_folder_path (str): Path to the input dataset folder.
        input_size (tuple): Target size of input images (e.g., (width, height)).

    Returns:
        tf.keras.preprocessing.image.DirectoryIterator: A dataset generator for model training.
    """
    datagen = create_image_generator(params)
    generator = datagen.flow_from_directory(
        input_folder_path,
        target_size=input_size,
        batch_size=params["batch_size"],
        class_mode='categorical'
    )
    return generator

def read_data_preparation_params(key):
    """
    Read data preparation parameters from a YAML file.

    Args:
        key (str): The key to identify the specific set of parameters in the YAML file.

    Returns:
        dict or None: A dictionary containing data preparation parameters if successful,
        or None if there was an error.
    """
    # Path of the parameters file
    params_path = Path("params.yaml")

    # Read data preparation parameters
    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params[key]
        except yaml.YAMLError as exc:
            print(exc)
            params = None
    return params
