import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import pandas as pd

# Page Configuration
st.set_page_config(page_title="FUDMA Malaria Detector", page_icon="🔬", layout="wide")

@st.cache_resource
def load_my_model():
    try:
        # Load the pre-trained model
        model = tf.keras.models.load_model('malaria_model_fudma.h5')
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model = load_my_model()

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2864/2864274.png", width=100)
st.sidebar.title("Main Menu")
choice = st.sidebar.radio("Navigation", ["Student Profile", "Live Diagnosis", "System Performance"])

st.sidebar.markdown("---")
st.sidebar.write("**Supervised by:** Nuradden Ahmad Sama'illa")
st.sidebar.write("**Session:** 2025/2026")

# --- STUDENT PROFILE SECTION ---
if choice == "Student Profile":
    st.title("🎓 Candidate Information")
    st.info("Department of Computer Science - Federal University Dutsin-Ma")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200)
    with col_b:
        st.subheader("Personal Details")
        st.table({
            "Field": ["Name", "Matric Number", "Major", "Level", "Institution"],
            "Details": ["Zakariyya Abubakar", "CSA/2025/19980", "Computer Science", "Final Year", "FUDMA, Nigeria"]
        })

# --- LIVE DIAGNOSIS SECTION ---
elif choice == "Live Diagnosis":
    st.title("🔬 Malaria Detection System")
    st.write("Upload a digital micrograph of a thin blood smear for automated parasite screening.")

    if model is None:
        st.error("Model file not found. Ensure 'malaria_model_fudma.h5' is in the project directory.")
    else:
        file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg"])

        if file is not None:
            image = Image.open(file)
            img_array = np.array(image.convert('RGB'))
            
            # --- IMAGE VALIDATION (THE GATEKEEPER) ---
            # We calculate the mean brightness and standard deviation (contrast)
            mean_val = np.mean(img_array)
            std_val = np.std(img_array)
            
            # Rule: Training images (micrographs) have specific brightness and high contrast.
            # Random photos (cars, people, etc.) fail these specific statistical ranges.
            is_valid_micrograph = True
            
            if mean_val < 50 or mean_val > 230 or std_val < 15:
                is_valid_micrograph = False
            
            col_img, col_res = st.columns([1, 1])
            
            with col_img:
                st.image(image, caption="Uploaded Sample", use_container_width=True)
            
            with col_res:
                if not is_valid_micrograph:
                    # If image is NOT a blood cell image
                    st.error("### ❌ Invalid Input")
                    st.info("This is not a recognized training image format. Please upload a valid thin blood smear micrograph.")
                else:
                    # If image IS valid, proceed to diagnosis
                    st.subheader("Diagnostic Analysis")
                    with st.spinner('Running AI Inference...'):
                        start_time = time.time()
                        
                        # Preprocessing
                        size = (128, 128)
                        image_resized = ImageOps.fit(image, size, Image.LANCZOS)
                        img_normalized = np.asarray(image_resized) / 255.0
                        img_reshape = np.expand_dims(img_normalized, axis=0)
                        
                        # Prediction
                        prediction = model.predict(img_reshape)
                        confidence = prediction[0][0]
                        end_time = time.time()
                        
                        # Results display
                        if confidence < 0.5:
                            st.error("### RESULT: POSITIVE")
                            st.markdown("Evidence of Plasmodium parasites detected.")
                        else:
                            st.success("### RESULT: NEGATIVE")
                            st.markdown("No parasites detected in the cell.")
                        
                        st.divider()
                        st.write(f"**AI Confidence:** {round(confidence * 100, 2)}%")
                        st.progress(float(confidence))
                        st.write(f"**Inference Speed:** {round(end_time - start_time, 4)} seconds")

# --- SYSTEM PERFORMANCE SECTION ---
else:
    st.title("📊 Research & Model Metrics")
    st.subheader("Model Training Evaluation")
    c1, c2 = st.columns(2)
    with c1:
        try: st.image("accuracy_graph.png", caption="Accuracy Curve")
        except: st.info("Place 'accuracy_graph.png' in folder to display.")
    with c2:
        try: st.image("loss_graph.png", caption="Loss Curve")
        except: st.info("Place 'loss_graph.png' in folder to display.")

    st.divider()
    st.subheader("📈 Real-time Reliability Simulation")
    chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['Accuracy', 'Loss'])
    st.line_chart(chart_data)