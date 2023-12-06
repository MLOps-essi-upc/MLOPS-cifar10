"""Main script for the fastAPI: API initialization and endpoints for the users."""

import io
from http import HTTPStatus
import pandas as pd

from fastapi import FastAPI, UploadFile, HTTPException, File, APIRouter
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import time
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

from src.app.classes import ImagesType
from src.models.utils import get_model

model = get_model()

# Define application
app = FastAPI(
    title="Cifar 10 Classifier",
    description="""This API facilitates the application of the InceptionV3 algorithm trained on the ImageNet dataset 
                    for making predictions with the Cifar 10 Classifier. Users can harness the power of InceptionV3, 
                    a state-of-the-art convolutional neural network, to achieve accurate and sophisticated image 
                    classification within the context of the Cifar 10 dataset.""",
    version="0.1",
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Use Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory="static")

# Create an APIRouter for your app
router = APIRouter()

# Endpoint to list possible labels for the image
@app.get("/list_labels")
async def list_labels():
    """
        Lists all the possible classification labels
    """

    label_names = [label.name for label in ImagesType]

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"classes": label_names}
    }


# Endpoint to predict the class of an image
@app.post("/prediction")
async def prediction(file: UploadFile = File(...)):
    """
        predicts the class of a given image
    """

    # Ensure the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    img = image.img_to_array(Image.open(io.BytesIO(content)).resize((75, 75)))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)/255.0
    st = time.time()
    prediction_probabilities = model.predict(img_array)[0]
    et = time.time()

    predicted_class = ImagesType(np.argmax(prediction_probabilities)).name
    result_dict = {label.name: float(prob)
                   for label, prob in zip(ImagesType, prediction_probabilities)}

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predicted_class": predicted_class, "probabilities": result_dict},
        "elapsed-time": round(et-st,4)
    }

@app.get("/", response_class=HTMLResponse)
async def get_upload_page():
    return FileResponse("static/index.html")