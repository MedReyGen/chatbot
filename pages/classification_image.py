import streamlit as st
from keras.models import load_model
from keras.models import model_from_json
import h5py
import json
from PIL import Image

from utils.util import classify, call_chatbot

st.title("MedReyGen 🫁")
st.subheader("Asisten Medis untuk Penyakit Pernapasan")
st.markdown("""
Asisten ini dapat klasifikasi gambar X-Ray apakah termasuk ke dalam penyakit pernapasan seperti pneumonia, tuberkulosis (TBC), dan COVID-19.
""")

# Upload file
file = st.file_uploader('Upload gambar JPEG, JPG, atau PNG', type=['jpeg', 'jpg', 'png'])

# Load file .h5
with h5py.File('./model/pneumonia_classifier.h5', 'r') as f:
    model_config = f.attrs.get('model_config')
    model_config = json.loads(model_config)

# Edit config: delete 'groups' from DepthwiseConv2D layers
for layer in model_config['config']['layers']:
    if layer['class_name'] == 'DepthwiseConv2D':
        layer['config'].pop('groups', None)

# Bangun model baru dari config yang sudah diperbaiki
# Read model after repairing the config
model = model_from_json(json.dumps(model_config))

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True)

    # Classify image
    class_name_result, conf_score = classify(image, model, class_names)

    # Write classification result
    st.write("### Hasil klasifikasi X-Ray adalah {}".format(class_name_result))
    st.write("Confidence score model {}".format(conf_score))

    # Initialize chat history
    if "classification_messages" not in st.session_state:
        st.session_state.classification_messages = []
    
    if "has_classified" not in st.session_state:
        st.session_state.has_classified = False

    if "pending_user_prompt" not in st.session_state:
        st.session_state.pending_user_prompt = None

    if "pending_assistant_response" not in st.session_state:
        st.session_state.pending_assistant_response = None

    if not st.session_state.has_classified:
        first_prompt = ""

        if(class_name_result.lower() == "normal"):
            first_prompt = "Saat ini hasil X-Ray ku normal. Apa yang harus kulakukan untuk menjaga kesehatan pernafasanku agar terhindar dari penyakit TBC, pneumonia, atau pun COVID-19?"
        elif(class_name_result.lower() in ["pneumonia", "covid-19", "tbc"]):
            first_prompt = f"Jelaskan mengenai penyakit {class_name_result}"
        else:
            first_prompt = "Jelaskan penyakit TBC, Covid-19, dan Pneumonia secara singkat"

        # Show conversation
        st.divider()
        st.write("### 🩺 Asisten Medis")

        st.session_state.classification_messages.append({"role": "user", "content": first_prompt})

        with st.chat_message("user"):
            st.markdown(first_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Memikirkan jawaban"):
                assistant_response = call_chatbot(first_prompt)
                st.markdown(assistant_response)
                st.session_state.classification_messages.append({"role": "assistant", "content": assistant_response})
        
        st.session_state.has_classified = True
    
    if st.session_state.has_classified:
        st.divider()
        st.write("### 🩺 Asisten Medis")

        prompt = st.chat_input("Tanyakan sesuatu lebih lanjut")

        # Render semua pesan sebelumnya, kecuali user prompt terakhir (kalau baru saja diketik)
        for i, message in enumerate(st.session_state.classification_messages):
            # Jika user baru saja input prompt, skip render user terakhir (supaya gak dobel)
            if prompt and i == len(st.session_state.classification_messages) - 1 and message["role"] == "user":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt:
            new_message = {"role": "user", "content": prompt}
            st.session_state.classification_messages.append(new_message)
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Memikirkan jawaban"):
                    assistant_response = call_chatbot(prompt, st.session_state.classification_messages)
                    st.markdown(assistant_response)
                    st.session_state.classification_messages.append({"role": "assistant", "content": assistant_response})