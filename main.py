import json
import os

import streamlit as st
import tensorflow as tf
import keras
from PIL import Image

try:
    import gdown  # type: ignore
except ImportError:
    gdown = None


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Citrus Plant Disease Detection",
    page_icon="🍊",
    layout="centered"
)


# ============================================================
# Model Configuration
# ============================================================

MODEL_PATH = "citrus_disease_model.keras"
FILE_ID = "1xbs3vuWIc97-pqwpYkRqUx23Ehf0xlcd"


# ============================================================
# Ensure Model Exists
# ============================================================

def ensure_model():
    """Download the model from Google Drive if it doesn't exist."""

    if os.path.exists(MODEL_PATH):
        return

    if gdown is None:
        raise ImportError(
            "gdown is required. Install it using: pip install gdown"
        )

    with st.spinner("Downloading AI model for the first time..."):
        downloaded_file = gdown.download(
            id=FILE_ID,
            output=MODEL_PATH,
            quiet=False
        )

    if downloaded_file is None or not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model download failed. Please check the Google Drive file."
        )


# ============================================================
# Load Model
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Download and load the model only when needed."""

    ensure_model()

    return keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# Load Classes
# ============================================================

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)


# ============================================================
# Disease Information
# ============================================================

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


# ============================================================
# Prediction Function
# ============================================================

def predict(image):
    """Predict citrus leaf disease."""

    # Load model only when prediction is requested
    model = load_model()

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = tf.keras.utils.img_to_array(image)
    img = tf.expand_dims(img, 0)
    img = img / 255.0

    prediction = model.predict(
        img,
        verbose=0
    )

    predicted_index = int(
        tf.argmax(prediction, axis=1)[0]
    )

    confidence = float(
        tf.reduce_max(prediction)
    ) * 100

    disease = class_indices[str(predicted_index)]

    description, treatment, prevention = disease_info.get(
        disease,
        (
            "Information is not available for this disease.",
            "Please consult an agricultural specialist.",
            "Continue regular monitoring."
        )
    )

    return (
        disease,
        confidence,
        description,
        treatment,
        prevention
    )


# ============================================================
# UI
# ============================================================

st.title("🍊 Citrus Plant Disease Detection")

st.write(
    "Upload a citrus leaf image and detect plant diseases "
    "using Deep Learning."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Image Preview + Prediction
# ============================================================

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease", type="primary"):

        with st.spinner(
            "Analyzing image... The first prediction may take a little longer."
        ):

            try:

                (
                    disease,
                    confidence,
                    description,
                    treatment,
                    prevention
                ) = predict(image)

                st.success(
                    f"Prediction: {disease}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    min(confidence / 100, 1.0)
                )

                st.subheader("Description")
                st.info(description)

                st.subheader("Treatment")
                st.warning(treatment)

                st.subheader("Prevention")
                st.success(prevention)

            except Exception as e:

                st.error(
                    "An error occurred while loading the model "
                    "or making the prediction."
                )

                st.exception(e)
