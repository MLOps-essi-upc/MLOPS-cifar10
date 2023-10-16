import yaml

from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

def create_image_generator(params):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Apply InceptionV3 preprocessing
        zoom_range=params["zoom_range"],
        horizontal_flip=params["horizontal_flip"]
    )
    return datagen

def load_dataset(params, input_folder_path, input_size):
    datagen = create_image_generator(params)
    train_generator = datagen.flow_from_directory(
        input_folder_path,
        target_size=input_size,
        batch_size=params["batch_size"],
        class_mode='categorical'
    )
    return train_generator

def readDataPreparationParams(key):
    # Path of the parameters file
    params_path = Path("params.yaml")
    
    # Read data preparation parameters
    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params[key]
        except yaml.YAMLError as exc:
            print(exc)
            params = None
    return params
