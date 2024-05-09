import os
import base64
import requests
import warnings

from io import BytesIO

import streamlit as st

warnings.filterwarnings("ignore")

##################

work_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(work_dir)

##################

URL = "http://0.0.0.0:7070"

##################

if __name__ == "__main__":
    st.sidebar.title("Select your use-case:")
    usecase_choice = st.sidebar.selectbox(
        "Select your use-case:",
        ("Text", "Image", "Audio")
    )

    st.sidebar.write("You selected:", usecase_choice)

    st.title(f"Inference for Use Case: {usecase_choice}")
    API_ENDPOINT = f"{URL}/{usecase_choice.lower()}"

    if usecase_choice == "Text":
        model_type = st.sidebar.selectbox(
            "Select the pretrained model:",
            ("LSTM",
             "Transformer")
        )

        user_input = st.text_input("Input a text to be predicted")
        if user_input:
            # hit the API
            response = requests.post(
                API_ENDPOINT,
                json={"model_type": model_type.lower(),
                      "user_input": user_input}
            )
            resp_json = response.json()
            pred = resp_json["pred"]
            proba = resp_json["proba"]

            # present the results to frontend
            st.write(f"Sentiment: {pred}-{'Positive' if pred == 1 else 'Negative'}")
            st.write("Score:", proba)

    elif usecase_choice == "Image":
        uploaded_img = st.file_uploader("Choose an image to predict")
        if uploaded_img:
            # encode the upload image
            bytesio = BytesIO(uploaded_img.getvalue())
            bytesio.seek(0)
            bytes_img = bytesio.read()
            base64_img = base64.b64encode(bytes_img).decode("utf-8")

            # hit the API
            response = requests.post(
                API_ENDPOINT, json={"img_bytes": base64_img}
            )
            resp_json = response.json()
            pred = resp_json["pred"]
            pred_label = resp_json["pred_label"]
            proba = resp_json["proba"]

            # present the results to frontend
            st.write(f"The predicted label: {pred}-{pred_label}")
            st.write("Probability:", proba)

    elif usecase_choice == "Audio":
        uploaded_audio = st.file_uploader("Choose an audio to predict")
        if uploaded_audio:
            # encode the upload audio
            bytesio = BytesIO(uploaded_audio.getvalue())
            bytesio.seek(0)
            bytes_audio = bytesio.read()
            base64_audio = base64.b64encode(bytes_audio).decode("utf-8")

            # hit the API
            response = requests.post(
                API_ENDPOINT, json={"audio_bytes": base64_audio}
            )
            resp_json = response.json()
            pred = resp_json["pred"]
            pred_label = resp_json["pred_label"]
            proba = resp_json["proba"]

            # present the results to frontend
            st.write(f"The predicted label: {pred}-{pred_label}")
            st.write("Probability:", proba)
