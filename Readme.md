# 🔬 AI-Powered Malaria Parasite Detection System

### Department of Computer Science
**Federal University Dutsin-Ma (FUDMA)**

This repository contains a Streamlit-based web application that utilizes Deep Learning (Convolutional Neural Networks) to detect malaria parasites in thin blood smear images. The system is designed to assist in rapid diagnostic screening using AI.

---

## 🚀 Features

*   **Automated Diagnosis:** Classifies blood cell images into **Parasitized** or **Uninfected**.
*   **Image Validation (Gatekeeper):** A built-in statistical validation layer that ensures only valid microscope micrographs are processed. It rejects unrelated images (e.g., selfies, objects, or low-quality photos).
*   **Performance Metrics:** Real-time tracking of inference speed and AI confidence scores.
*   **Research Dashboard:** Visualizes model training history, including accuracy and loss curves.
*   **Professional UI:** A clean, multi-page interface built with Streamlit.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.9+
*   **Framework:** [Streamlit](https://streamlit.io/)
*   **Deep Learning:** [TensorFlow / Keras](https://www.tensorflow.org/)
*   **Image Processing:** Pillow (PIL), NumPy
*   **Data Handling:** Pandas

---

## 📂 Project Structure


## Data set from:
    https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria


```text
├── app.py                      # Main Streamlit application script
├── malaria_model_fudma.h5       # Pre-trained CNN model
├── requirements.txt            # Python dependencies
├── accuracy_graph.png          # Model training accuracy plot
├── loss_graph.png              # Model training loss plot
└── README.md                   # Project documentation
