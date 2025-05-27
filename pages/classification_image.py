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

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if file is not None:
    if file.name != st.session_state.uploaded_filename:
        st.session_state.clear()
        st.session_state.uploaded_filename = file.name

@st.cache_resource
def load_assets():
    model = load_model('./model/x_ray_classifier.h5')

    with open('./model/labels.txt', 'r') as f:
        class_names = [a.strip().split(' ')[1] for a in f.readlines()]
    return model, class_names

model, class_names = load_assets()

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
    
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = None

    if "has_rendered_first_prompt" not in st.session_state:
        st.session_state.has_rendered_first_prompt = False
    
    # if "chat_input_buffer" not in st.session_state:
    #     st.session_state.chat_input_buffer = None
    
    # Show conversation
    st.divider()
    st.write("### 🩺 Asisten Medis")

    if not st.session_state.has_classified:
        first_prompt = ""

        if(class_name_result.lower() == "normal"):
            first_prompt = "Saat ini hasil X-Ray ku normal. Apa yang harus kulakukan untuk menjaga kesehatan pernafasanku agar terhindar dari penyakit TBC, pneumonia, atau pun COVID-19?"
        elif(class_name_result.lower() in ["pneumonia", "covid", "tbc"]):
            first_prompt = f"Jelaskan mengenai penyakit {class_name_result} dan bagaimana cara menanganinya?"
        else:
            first_prompt = "Jelaskan penyakit TBC, Covid-19, dan Pneumonia secara singkat"

        st.session_state.classification_messages.append({"role": "user", "content": first_prompt})

        with st.chat_message("user"):
            st.markdown(first_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Memikirkan jawaban"):
                assistant_response = call_chatbot(first_prompt)
                st.markdown(assistant_response)
        st.session_state.classification_messages.append({"role": "assistant", "content": assistant_response})
        
        st.session_state.has_classified = True
        st.session_state.has_rendered_first_prompt = True
    
    else:
        for i, message in enumerate(st.session_state.classification_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Tanyakan sesuatu lebih lanjut")

    if prompt and  prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        # prompt = st.session_state.chat_input_buffer

        # if st.session_state.get("last_prompt") != prompt:
        #     st.session_state.last_prompt = prompt
        #     new_message = {"role": "user", "content": prompt}
        #     st.session_state.classification_messages.append(new_message)
        #     with st.chat_message("user"):
        #         st.markdown(prompt)
            
        #     with st.chat_message("assistant"):
        #         with st.spinner("Memikirkan jawaban"):
        #             assistant_response = call_chatbot(prompt, st.session_state.classification_messages)
        #             st.markdown(assistant_response)
        #     st.session_state.classification_messages.append({"role": "assistant", "content": assistant_response})
        
        # st.session_state.classification_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.classification_messages.append({"role": "user", "content": prompt})
            
        with st.chat_message("assistant"):
            with st.spinner("Memikirkan jawaban"):
                assistant_response = call_chatbot(prompt, st.session_state.classification_messages)
                st.markdown(assistant_response)

        st.session_state.classification_messages.append({"role": "assistant", "content": assistant_response})
        # st.session_state.chat_input_buffer = None

        print(st.session_state.classification_messages)
        print()
        # print()