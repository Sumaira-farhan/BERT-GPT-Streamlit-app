import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("BERT and GPT AI Application")

menu = st.sidebar.selectbox(
    "Choose Task",
    ["Sentiment Analysis", "Text Generation"]
)

# ---------------- SENTIMENT ---------------- #

if menu == "Sentiment Analysis":

    st.header("BERT Sentiment Analysis")

    text = st.text_area("Enter text")

    if st.button("Analyze"):

        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text}
        )

        result = response.json()

        st.success(f"Prediction: {result['label']}")
        st.write(f"Confidence: {result['score']}")

# ---------------- GPT ---------------- #

elif menu == "Text Generation":

    st.header("GPT Text Generation")

    prompt = st.text_area("Enter prompt")

    if st.button("Generate"):

        response = requests.post(
            f"{API_URL}/generate",
            json={"prompt": prompt}
        )

        result = response.json()

        st.write(result["generated_text"])
