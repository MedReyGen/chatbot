from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Google AI client
client = genai.Client(
    vertexai=True,
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION"),
)

@app.route('/generate', methods=['POST'])
def generate_response():
    # Get user query from request
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    si_text1 = """anda adalah asisten virtual dalam memberikan informasi, saran medis awal, dan panduan lanjutan kepada pengguna untuk penyakit pernafasan seperti pneumonia, tuberkulosis (TBC), dan COVID-19. tidak bisa menjawab penyakit lainnya."""

    model = "gemini-2.5-pro-preview-05-06"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_query)
            ]
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.88,
        seed=0,
        max_output_tokens=3000,
        safety_settings=[types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        system_instruction=[types.Part.from_text(text=si_text1)],
    )

    # Generate response (non-streaming for API)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)