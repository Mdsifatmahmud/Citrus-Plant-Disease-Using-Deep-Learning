# 🍊 Citrus Plant Disease Using Deep Learning

A deep learning-based web application for detecting citrus plant diseases from leaf images. The system helps farmers, researchers, and agricultural professionals identify diseases quickly by uploading an image of a citrus leaf. It uses a trained Convolutional Neural Network (CNN) model and provides real-time predictions through an interactive Streamlit interface.

---

## 📌 Features

* 🌿 Detects citrus plant diseases from leaf images
* 🤖 Deep learning-based image classification
* 📤 Simple image upload interface
* ⚡ Real-time prediction
* 📊 Displays predicted disease with confidence score
* 💻 User-friendly Streamlit web application
* ☁️ Ready for deployment on Streamlit Cloud

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* OpenCV
* NumPy
* Pillow
* Matplotlib

---

## 📂 Project Structure

```text
Citrus-Plant-Disease-Using-Deep-Learning/
│
├── app.py
├── requirements.txt
├── citrus_disease_model.h5
├── assets/
├── images/
├── utils.py
├── README.md
└── ...
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Citrus-Plant-Disease-Using-Deep-Learning.git
```

### 2. Navigate to the Project

```bash
cd Citrus-Plant-Disease-Using-Deep-Learning
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

After running the command, open the local URL displayed in the terminal.

---

## 📷 How to Use

1. Launch the application.
2. Upload a citrus leaf image.
3. Click **Predict** (if applicable).
4. The model analyzes the image.
5. View the predicted disease and confidence score.

---

## 🧠 Model Information

* Model Type: Convolutional Neural Network (CNN)
* Framework: TensorFlow / Keras
* Input Image Size: **224 × 224**
* Output: Citrus disease classification

---

## 📊 Supported Disease Classes

The model can classify multiple citrus leaf diseases depending on the trained dataset, such as:

* Black Spot
* Canker
* Greening
* Healthy
* Melanose
* Scab
* Other trained classes

> **Note:** The available classes depend on the dataset used during training.

---

## 📸 Sample Workflow

```text
Leaf Image
      │
      ▼
Image Preprocessing
      │
      ▼
CNN Model Prediction
      │
      ▼
Disease Classification
      │
      ▼
Prediction & Confidence Score
```

---

## 📦 Requirements

Example dependencies:

```text
streamlit
tensorflow
opencv-python
numpy
pillow
matplotlib
scikit-learn
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

* Support more citrus diseases
* Disease severity estimation
* Treatment and fertilizer recommendations
* Mobile-friendly interface
* Model optimization for faster inference
* Multi-language support

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Md. Sifat Mahmud**

* GitHub: https://github.com/Mdsifatmahmud

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub. It helps others discover the project and motivates future development.
