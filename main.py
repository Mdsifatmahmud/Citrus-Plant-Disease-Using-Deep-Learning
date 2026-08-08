import json
import os

import numpy as np
import streamlit as st
from PIL import Image

from ai_edge_litert.interpreter import Interpreter

try:
    import gdown
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

MODEL_PATH = "citrus_disease_model.tflite"

# Replace this with your NEW TFLite Google Drive file ID
FILE_ID = "1gQ8lGsis3kt2FipvpSF73xd2n3jtxvoK"


# ============================================================
# Download Model
# ============================================================

def ensure_model():
    """Download TFLite model from Google Drive if needed."""

    if os.path.exists(MODEL_PATH):
        return

    if gdown is None:
        raise ImportError(
            "gdown is required. Install it using: pip install gdown"
        )

    with st.spinner(
        "Downloading AI model for the first time..."
    ):

        downloaded_file = gdown.download(
            id=FILE_ID,
            output=MODEL_PATH,
            quiet=False
        )

    if (
        downloaded_file is None
        or not os.path.exists(MODEL_PATH)
    ):
        raise RuntimeError(
            "Model download failed. "
            "Please check the Google Drive file."
        )


# ============================================================
# Load TFLite Model
# ============================================================

@st.cache_resource(show_spinner=False)
def load_interpreter():

    ensure_model()

    interpreter = Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return interpreter


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

    interpreter = load_interpreter()

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert to NumPy array
    img = np.asarray(image, dtype=np.float32)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Get input/output information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    # Make prediction
    interpreter.set_tensor(
        input_index,
        img
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_index
    )

    # Get predicted class
    predicted_index = int(
        np.argmax(prediction[0])
    )

    # Get confidence
    confidence = float(
        np.max(prediction[0])
    ) * 100

    # Get disease name
    disease = class_indices[str(predicted_index)]

    # Disease information
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


# ============================================================
# Upload Image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Predict Disease",
        type="primary"
    ):

        with st.spinner(
            "Analyzing image..."
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
                    "An error occurred while making "
                    "the prediction."
                )

                st.exception(e)