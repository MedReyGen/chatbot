import streamlit as st
import requests
import json
import time

from utils.util import call_chatbot

st.title("MedReyGen 🫁")
st.subheader("Asisten Medis untuk Penyakit Pernapasan")
st.markdown("""
Asisten virtual untuk memberikan informasi, saran medis awal, dan panduan 
lanjutan mengenai penyakit pernapasan seperti pneumonia, tuberkulosis (TBC), dan COVID-19.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_prompt_for_chatbot_only" not in st.session_state:
    st.session_state.last_prompt_for_chatbot_only = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Tanyakan sesuatu tentang penyakit pernapasan...")

# Backend URL
BACKEND_URL = "http://localhost:5000/generate"

if prompt and prompt != st.session_state.last_prompt_for_chatbot_only:
    st.session_state.last_prompt_for_chatbot_only = prompt

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show assistant response with a spinner while loading
    # with st.chat_message("assistant"):
    #     with st.spinner("Memikirkan jawaban..."):
    #         assistant_response = call_chatbot(prompt, st.session_state.messages)
    #         st.markdown(assistant_response)

    # with st.chat_message("assistant"):
    #     placeholder = st.empty()
    #     with st.spinner("Memikirkan jawaban"):
    #         assistant_response = call_chatbot(prompt, st.session_state.messages)
    #     placeholder.markdown(assistant_response)

    # with st.chat_message("assistant"):
    #     placeholder = st.empty()
    #     with st.spinner("Memikirkan jawaban..."):
    #         response_text = ""
    #         for partial in call_chatbot(prompt, st.session_state.messages):
    #             response_text = partial
    #             placeholder.markdown(response_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream = call_chatbot(prompt, st.session_state.messages)

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

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )