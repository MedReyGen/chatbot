import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

import requests

def classify(image, model, class_names):
    """test"""
    # Convert image to (224, 224)
    image_converted = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image_converted)

    # Normalize image supaya rentang nilainya 0 hingga 1 karena gambar dinilai RGB
    normalize_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Set model input
    # This means 1 image with 224,224 size and 3 channels of color
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalize_image_array

    # Make prediction
    prediction_result = model.predict(data)
    index = np.argmax(prediction_result)
    class_name = class_names[index]
    confidence_score = prediction_result[0][index]

    return class_name, confidence_score

def call_chatbot(message, context=None):
    BACKEND_URL = "http://localhost:5000/generate"
    try:
        full_context = context if context else []
        full_context.append({
            "role": "user",
            "content": message
        })
        response = requests.post(
            BACKEND_URL,
            json={"query": full_context},
            headers={"Content-Type": "application/json"}
        )

        response.raise_for_status()
        return response.json().get("response", "Maaf, tidak ada respon mengenai hal ini")
    
    except Exception as e:
        return f"Error: {str(e)}"