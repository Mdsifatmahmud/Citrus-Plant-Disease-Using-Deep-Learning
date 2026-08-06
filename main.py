import json
import os

try:
    import gdown  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    gdown = None

import streamlit as st
import tensorflow as tf
import keras
from PIL import Image

MODEL_PATH = "citrus_disease_model.keras"
FILE_ID = "1xbs3vuWIc97-pqwpYkRqUx23Ehf0xlcd"

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return

    if gdown is None:
        raise ImportError(
            "gdown is required. Install it using: pip install gdown"
        )

    gdown.download(
        id=FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )


ensure_model()
# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Citrus Plant Disease Detection",
    page_icon="🍊",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

model = load_model()

# ----------------------------
# Load Classes
# ----------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# ----------------------------
# Disease Information
# ----------------------------
disease_info = {
    "Anthracnose": (
        "Fungal disease causing dark lesions on leaves.",
        "Apply fungicide and prune infected leaves.",
        "Maintain good airflow and avoid excess moisture."
    ),
    "Bacterial Blight": (
        "Bacterial infection causing leaf spots.",
        "Use copper-based bactericide.",
        "Use disease-free seedlings."
    ),
    "Citrus Canker": (
        "Highly contagious bacterial disease.",
        "Remove infected leaves and apply copper spray.",
        "Avoid moving infected plants."
    ),
    "Curl Virus": (
        "Virus causing curled leaves.",
        "No direct cure available.",
        "Control insect vectors."
    ),
    "Deficiency Leaf": (
        "Leaf nutrient deficiency.",
        "Apply balanced fertilizer.",
        "Maintain soil nutrition."
    ),
    "Dry Leaf": (
        "Leaf drying due to stress or disease.",
        "Improve watering.",
        "Maintain proper irrigation."
    ),
    "Healthy Leaf": (
        "Leaf is healthy.",
        "No treatment required.",
        "Continue regular monitoring."
    ),
    "Sooty Mould": (
        "Black fungal growth caused by insect honeydew.",
        "Control insects and wash leaves.",
        "Regular pest control."
    ),
    "Spider Mites": (
        "Tiny mites damaging leaves.",
        "Use miticide.",
        "Maintain humidity and monitor plants."
    )
}

# ----------------------------
# Prediction Function
# ----------------------------
def predict(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = tf.keras.utils.img_to_array(image)
    img = tf.expand_dims(img, 0)
    img = img / 255.0

    prediction = model.predict(img, verbose=0)

    predicted_index = int(tf.argmax(prediction, axis=1)[0])
    confidence = float(tf.reduce_max(prediction)) * 100

    disease = class_indices[str(predicted_index)]

    description, treatment, prevention = disease_info[disease]

    return disease, confidence, description, treatment, prevention


# ----------------------------
# UI
# ----------------------------
st.title("🍊 Citrus Plant Disease Detection")

st.write(
    "Upload a citrus leaf image and detect plant diseases using Deep Learning."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):

        with st.spinner("Analyzing..."):

            disease, confidence, description, treatment, prevention = predict(image)

        st.success(f"Prediction: {disease}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(min(confidence / 100, 1.0))

        st.subheader("Description")
        st.info(description)

        st.subheader("Treatment")
        st.warning(treatment)

        st.subheader("Prevention")
        st.success(prevention)