import streamlit as st

chatbot_image_classification = st.Page(
    page="pages/classification_image.py",
    title="Klasifikasi Gambar X-Ray",
    icon=":material/category:",
    default=True
)

chatbot_page = st.Page(
    page="pages/chatbot.py",
    title="Chatbot dengan Vertex AI",
    icon=":material/chat:"
)

pg = st.navigation([chatbot_image_classification, chatbot_page])

st.logo("assets/laskar_ai_logo.png")
st.sidebar.text("Made with ❤️ by Grouper")

pg.run()