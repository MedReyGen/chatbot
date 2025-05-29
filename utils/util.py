import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

from tensorflow.keras.preprocessing import image

import requests

def classify(image_imported, model, class_names):
    """test"""
    # Convert image to (224, 224)
    image_converted = ImageOps.fit(image_imported, (224, 224), Image.Resampling.LANCZOS)

     # Convert image to numpy array
    image_array = image.img_to_array(image_converted)

    # Normalize image supaya rentang nilainya 0 hingga 1 karena gambar dinilai RGB
    normalize_image_array = image_array / 255.0

    # Set model input
    data = np.expand_dims(normalize_image_array, axis=0)

    # Make prediction
    prediction_result = model.predict(data)
    index = np.argmax(prediction_result, axis=1)[0]
    class_name = class_names[index]
    confidence_score = prediction_result[0][index]

    return class_name, confidence_score

def call_chatbot(message, context=None):
    BACKEND_STREAM_URL = "http://localhost:5000/generate-stream"

    full_context = context.copy() if context else []
    full_context.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            BACKEND_STREAM_URL,
            json={"query": full_context},
            headers={"Content-Type": "application/json"},
            stream=True
        )
        resp.raise_for_status()

        buffer = ""
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            text = chunk.decode("utf-8")
            buffer += text
            yield buffer

    except Exception as e:
        yield f"Error: {str(e)}"