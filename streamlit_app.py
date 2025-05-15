import streamlit as st
import requests
import json

st.set_page_config(
    page_title="MedReyGen - Medical Respiratory Assistant",
    page_icon="🫁",
    layout="centered"
)

st.title("MedReyGen 🫁")
st.subheader("Asisten Medis untuk Penyakit Pernapasan")
st.markdown("""
Asisten virtual untuk memberikan informasi, saran medis awal, dan panduan 
lanjutan mengenai penyakit pernapasan seperti pneumonia, tuberkulosis (TBC), dan COVID-19.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Tanyakan sesuatu tentang penyakit pernapasan...")

# Backend URL
BACKEND_URL = "http://localhost:5000/generate"

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show assistant response with a spinner while loading
    with st.chat_message("assistant"):
        with st.spinner("Memikirkan jawaban..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"query": prompt},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()  # Raise exception for HTTP errors
                
                response_data = response.json()
                assistant_response = response_data.get("response", "Maaf, terjadi kesalahan.")
                
                # Display the response
                st.markdown(assistant_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            except Exception as e:
                st.error(f"Error: {str(e)}")