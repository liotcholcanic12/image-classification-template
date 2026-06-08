from tensorflow.keras.models import load_model
from PIL import Image
import gradio as gr
import numpy as np
import os

MODEL_PATH = "saved_model/model.h5"

IMG_HEIGHT = 224
IMG_WIDTH = 224

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\nRun train.py first."
    )

model = load_model(MODEL_PATH)

DATASET_PATH = "dataset"

CLASS_NAMES = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(
            DATASET_PATH,
            folder
        )
    )
])

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (
            IMG_WIDTH,
            IMG_HEIGHT
        )
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array

def classify_image(image):

    image_array = preprocess_image(
        image
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        prediction
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = prediction[
        predicted_index
    ]

    result = (
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}\n\n"
    )

    result += "Class Probabilities:\n"

    for cls, prob in zip(
        CLASS_NAMES,
        prediction
    ):
        result += (
            f"{cls}: {prob:.2%}\n"
        )

    return result

with gr.Blocks() as app:

    gr.Markdown(
        "# Image Classification Template"
    )

    gr.Markdown(
        "Upload an image to classify."
    )

    image_input = gr.Image(
        type="pil"
    )

    output = gr.Textbox(
        lines=10
    )

    predict_button = gr.Button(
        "Predict"
    )

    predict_button.click(
        classify_image,
        inputs=image_input,
        outputs=output
    )

app.launch()
