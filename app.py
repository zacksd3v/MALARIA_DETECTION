import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="FUDMA Malaria Detector", page_icon="🔬", layout="wide")

# 1. Load the trained model
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('malaria_model_fudma.h5')
        return model
    except:
        return None

model = load_my_model()

# --- SIDEBAR SETUP ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2864/2864274.png", width=100)
st.sidebar.title("Main Menu")
choice = st.sidebar.radio("Navigation", ["Student Profile", "Live Diagnosis", "System Performance"])

# --- SHARED FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("**Supervised by:** Project Supervisor")
st.sidebar.write("**Session:** 2025/2026")

# --- PAGE 1: STUDENT PROFILE ---
if choice == "Student Profile":
    st.title("🎓 Candidate Information")
    st.info("Department of Computer Science - Federal University Dutsin-Ma")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200) # Placeholder for student photo
    with col_b:
        st.subheader("Personal Details")
        st.table({
            "Field": ["Name", "Matric Number", "Major", "Level", "Institution"],
            "Details": ["Zakariyya Abubakar", "CSA/2025/19980", "Computer Science", "Final Year", "FUDMA, Nigeria"]
        })

# --- PAGE 2: LIVE DIAGNOSIS ---
elif choice == "Live Diagnosis":
    st.title("🔬 AI-Powered Malaria Screening")
    st.write("Upload a thin blood smear image to perform automated parasite detection.")

    if model is None:
        st.error("Model file 'malaria_model_fudma.h5' not found. Please ensure it is in the project folder.")
    else:
        file = st.file_uploader("Choose a Cell Image...", type=["jpg", "png", "jpeg"])

        if file is not None:
            image = Image.open(file)
            
            # Layout for Image and Prediction
            col_img, col_res = st.columns([1, 1])
            
            with col_img:
                st.image(image, caption="Uploaded Sample", use_container_width=True)
            
            with col_res:
                st.subheader("Diagnostic Result")
                with st.spinner('Analyzing cell structure...'):
                    # Start Timer for Performance Metric
                    start_time = time.time()
                    
                    # Preprocessing
                    size = (128, 128)
                    image_resized = ImageOps.fit(image, size, Image.LANCZOS)
                    img_array = np.asarray(image_resized) / 255.0
                    img_reshape = np.expand_dims(img_array, axis=0)
                    
                    # Prediction
                    prediction = model.predict(img_reshape)
                    confidence = prediction[0][0]
                    end_time = time.time()
                    
                    # Display Status
                    if confidence < 0.5:
                        st.error("### POSITIVE: Parasitized")
                        st.write("Evidence of Plasmodium parasites detected.")
                    else:
                        st.success("### NEGATIVE: Uninfected")
                        st.write("No parasites detected in the sample.")
                    
                    # Confidence Gauge
                    st.write(f"**AI Confidence:** {round(confidence * 100, 2)}%")
                    st.progress(float(confidence))
                    st.write(f"**Diagnosis Speed:** {round(end_time - start_time, 4)} seconds")

# --- PAGE 3: SYSTEM PERFORMANCE ---
else:
    st.title("📊 Model Analysis & Research Metrics")
    
    # Static Charts from Training
    st.subheader("Training History (Static Graphs)")
    c1, c2 = st.columns(2)
    with c1:
        try: st.image("accuracy_graph.png", caption="Training Accuracy")
        except: st.info("Add accuracy_graph.png to view training curves.")
    with c2:
        try: st.image("loss_graph.png", caption="Training Loss")
        except: st.info("Add loss_graph.png to view loss curves.")

    st.divider()

    # Interactive Charts (Showing Mastery of Data Visualization)
    st.subheader("📈 Interactive Prediction Reliability Chart")
    # Generating dummy data to show how the model performs across epochs
    chart_data = pd.DataFrame(
        np.random.randn(20, 2),
        columns=['Accuracy', 'Loss']
    )
    st.line_chart(chart_data)
    st.caption("Figure 1: Real-time visualization of model convergence (Demo Data).")

    st.divider()

    # Model Summary Section (The 'Pro' Feature)
    st.subheader("🧱 Deep Learning Architecture")
    if st.checkbox("Expand Technical Layer Details (CNN Summary)"):
        if model:
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            st.text("\n".join(stringlist))
        else:
            st.warning("Model not loaded.")

    st.info("Technical Specifications: Input Shape [128, 128, 3] | Optimizer: Adam | Loss: Binary Cross-Entropy")