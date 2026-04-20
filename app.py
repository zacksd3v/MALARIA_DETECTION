import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time

# 1. Loda model 
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('malaria_model_fudma.h5')
    return model

model = load_my_model()

# --- SIDEBAR SETUP ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2864/2864274.png", width=100)
st.sidebar.title("Dashboard")
choice = st.sidebar.selectbox("Zabi Sashe:", ["Gwajin Hoto (Detection)", "Binciken Model (Analysis)"])

# --- PAGE 1: DETECTION ---
if choice == "Gwajin Hoto (Detection)":
    st.title("🔬 Smart Malaria Diagnostic System")
    st.write("Wannan tsarin yana amfani da AI don gano cutar Malaria cikin sauri.")
    
    file = st.file_uploader("Loda hoton jini (Cell Image)...", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Hoton da aka loda", use_column_width=True)
        
        # Fara kirga lokaci
        start_time = time.time()
        
        # Preprocessing
        size = (128, 128)
        image_resized = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = np.asarray(image_resized) / 255.0
        img_reshape = np.expand_dims(img_array, axis=0)
        
        # Hasali (Prediction)
        prediction = model.predict(img_reshape)
        confidence = prediction[0][0]
        
        end_time = time.time()
        
        st.divider()
        
        # Nuna Sakamako
        col1, col2 = st.columns(2)
        
        with col1:
            if confidence > 0.5:
                st.error("### SAKAMAKO: PARASITIZED")
                st.write("An samu kwayar cutar malaria a wannan jinin.")
            else:
                st.success("### SAKAMAKO: UNINFECTED")
                st.write("Wannan jinin ba shi da kwayar cutar malaria.")
        
        with col2:
            st.write(f"**Confidence Score:** {round(confidence * 100, 2)}%")
            st.progress(float(confidence)) # Gauge bar
            st.write(f"**Processing Time:** {round(end_time - start_time, 4)}s")

# --- PAGE 2: ANALYSIS ---
else:
    st.title("📊 Model Performance & Architecture")
    
    # Sashen Graphs
    st.subheader("Training Statistics")
    col3, col4 = st.columns(2)
    
    with col3:
        try:
            st.image("accuracy_graph.png", caption="Model Accuracy Plot")
        except:
            st.warning("Bamu samu hoton accuracy_graph.png ba.")
            
    with col4:
        try:
            st.image("loss_graph.png", caption="Model Loss Plot")
        except:
            st.warning("Bamu samu hoton loss_graph.png ba.")

    # Sashen Model Summary (Wanda zai ba Supervisor sha'awa)
    st.divider()
    st.subheader("Model Architecture (CNN Layers)")
    if st.checkbox("Nuna Layer Details"):
        # Wannan zai dauko bayanan yadda ka gina model din
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        st.text(short_model_summary)
        
    st.info("Wannan model din an gina shi ne da Convolutional Neural Networks (CNN) don gano sigogin cuta.")