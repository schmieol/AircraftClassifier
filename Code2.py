import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os

# ==========================================
# Streamlit Page Configuration
# ==========================================

st.set_page_config(
    page_title="Military Aircraft Detection",
    layout="wide"
)

st.title("🛩️ Military Aircraft Object Detection")

st.write(
    "Upload images or videos of military aircraft. "
    "The AI model will automatically detect aircraft "
    "and display bounding boxes in real time."
)

# ==========================================
# Load YOLO Model
# ==========================================

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")

except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ==========================================
# File Upload
# ==========================================

uploaded_file = st.file_uploader(
    "Choose an image or video...",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)

# ==========================================
# Processing
# ==========================================

if uploaded_file is not None:

    file_extension = uploaded_file.name.split(".")[-1].lower()

    # ======================================
    # IMAGE PROCESSING
    # ======================================

    if file_extension in ["jpg", "jpeg", "png"]:

        image = Image.open(uploaded_file).convert("RGB")

        st.write("🔍 Processing image...")

        # YOLO prediction
        results = model(image)

        # Draw bounding boxes
        plotted = results[0].plot()

        # Convert BGR to RGB
        rgb = plotted[:, :, ::-1]

        # Display images
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                rgb,
                caption="Detection Result",
                use_container_width=True
            )

    # ======================================
    # VIDEO PROCESSING
    # ======================================

    elif file_extension in ["mp4", "avi"]:

        st.write("🎥 Processing video...")

        # Save uploaded video temporarily
        temp_file = tempfile.NamedTemporaryFile(
            delete=False
        )

        temp_file.write(uploaded_file.read())

        # Open video with OpenCV
        video = cv2.VideoCapture(temp_file.name)

        # Placeholder for live video frames
        stframe = st.empty()

        while video.isOpened():

            success, frame = video.read()

            if not success:
                break

            # YOLO inference
            results = model(frame)

            # Draw bounding boxes
            plotted = results[0].plot()

            # Convert BGR to RGB
            rgb = plotted[:, :, ::-1]

            # Display frame
            stframe.image(
                rgb,
                channels="RGB",
                use_container_width=True
            )

        # Release video
        video.release()

        # Remove temporary file
        os.remove(temp_file.name)

        st.success("✅ Video processing completed!")
