import streamlit as st
from keras.models import load_model
import h5py
import json
from PIL import Image
import time

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

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if file is not None:
    if file.name != st.session_state.uploaded_filename:
        # st.session_state.clear()
        for key in [
            "classification_messages",
            "has_classified",
            "last_prompt", 
            "has_rendered_first_prompt", 
            "uploaded_filename", 
            "uploaded_file"
        ]:
            st.session_state.pop(key, None)
        st.session_state.uploaded_filename = file.name
        st.session_state.uploaded_file = file
# else:
#     st.session_state.uploaded_filename = None
#     st.session_state.uploaded_file = None

@st.cache_resource
def load_assets():
    model = load_model('./model/x_ray_classifier_with_negative_data.h5')

    with open('./model/labels_with_negative_data.txt', 'r') as f:
        class_names = [a.strip().split(' ')[1] for a in f.readlines()]
    return model, class_names

model, class_names = load_assets()

image_file = file if file is not None else st.session_state.uploaded_file

# Display image
if file is not None or image_file is not None:
    image = Image.open(file if file is not None else image_file).convert('RGB')
    st.image(image, use_container_width=True)

    # Classify image
    class_name_result, conf_score = classify(image, model, class_names)

    percent = round(conf_score * 100)

    # Write classification result
    # st.write("### Hasil klasifikasi X-Ray adalah {}".format(class_name_result))

    if class_name_result.lower() == 'normal':
        result_text = f"""
            <div style='font-size: 20px;'>
                Berdasarkan analisis sistem, terdapat <b style='color:green; font-weight: 600; font-size: 28px;'>{percent}% kemungkinan</b> hasil X-Ray Anda <span style='color:green; font-weight: 600; font-size: 28px;'><b>normal</b></span>.
            </div>
        """
    elif class_name_result.lower() == 'none':
        result_text = f"""
            <div style='font-size: 20px;'>
                Berdasarkan analisis sistem, <b style='color:red; font-weight: 600; font-size: 28px;'>{percent}% kemungkinan gambar yang Anda upload bukan X-Rey dada.</b>
            </div>
        """
    elif class_name_result.lower() in ['tbc', 'covid', 'pneumonia']:
        result_text = f"""
            <div style='font-size: 20px;'>
                Berdasarkan analisis sistem, terdapat <b style='color:red; font-weight: 600; font-size: 28px;'>{percent}% kemungkinan</b> gambar X-Ray Anda memiliki ciri-ciri yang menyerupai
                <span style='color:red; font-weight: 600; font-size: 28px;'><b>{class_name_result.upper()}</b></span>.
            </div>
        """
    else:
        result_text = f"""
            <div style='font-size: 20px;'>
                Maaf, terdapat kesalahan pada sistem. Silahkan coba beberapa saat kemudian.
            </div>
        """
    
    st.markdown(result_text, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:16px; color:gray;'>"
            "Catatan: Analisis ini dilakukan oleh sistem AI berdasarkan gambar X-Ray, dan <b>bukan merupakan diagnosis medis</b>. "
            "Silakan konsultasikan lebih lanjut dengan tenaga medis profesional."
        "</div>",
        unsafe_allow_html=True
    )

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
            first_prompt = f"Jelaskan mengenai penyakit {class_name_result} secara singkat dan bagaimana cara menanganinya?"
        else:
            first_prompt = "Jelaskan penyakit TBC, Covid-19, dan Pneumonia secara singkat"

        st.session_state.classification_messages.append({"role": "user", "content": first_prompt})

        with st.chat_message("user"):
            st.markdown(first_prompt)

        # with st.chat_message("assistant"):
        #     with st.spinner("Memikirkan jawaban"):
        #         assistant_response = call_chatbot(first_prompt)
        #         st.markdown(assistant_response)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream = call_chatbot(first_prompt)

            try:
                with st.spinner("Memikirkan jawaban..."):
                    first_chunk = next(stream)
            except StopIteration:
                first_chunk = ""
            
            response_text = first_chunk
            placeholder.markdown(response_text)

            for partial in stream:
                response_text = partial
                placeholder.markdown(response_text)
                time.sleep(0.005)
        
        st.session_state.classification_messages.append({"role": "assistant", "content": response_text})
        
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
            
        # with st.chat_message("assistant"):
        #     with st.spinner("Memikirkan jawaban"):
        #         assistant_response = call_chatbot(prompt, st.session_state.classification_messages)
        #     st.markdown(assistant_response)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream = call_chatbot(prompt, st.session_state.classification_messages)

            try:
                with st.spinner("Memikirkan jawaban..."):
                    first_chunk = next(stream)
            except StopIteration:
                first_chunk = ""
            
            response_text = first_chunk
            placeholder.markdown(response_text)

            for partial in stream:
                response_text = partial
                placeholder.markdown(response_text)
                time.sleep(0.005)

        st.session_state.classification_messages.append({"role": "assistant", "content": response_text})
        # st.session_state.chat_input_buffer = None

        print(st.session_state.classification_messages)
        print()
        # print()