import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Military Aircraft Detection", layout="wide")
st.title("🛩️ Military Aircraft Object Detection")
st.write("Unggah gambar atau video pesawat militer. Model AI akan memproses dan menampilkan bounding box secara langsung.")

# 2. Memuat Model
@st.cache_resource
def load_model():
    model = YOLO('best.pt') 
    return model

try:
    model = load_model()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Antarmuka Unggah File (Sekarang mendukung mp4 dan avi)
uploaded_file = st.file_uploader("Pilih gambar atau video...", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    # Mendapatkan ekstensi file untuk menentukan jenis pemrosesan
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # ==========================================
    # LOGIKA PEMROSESAN GAMBAR
    # ==========================================
    if file_extension in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file).convert("RGB")
        st.write("Sedang memproses gambar...")
        
        results = model(image)
        res_plotted = results[0].plot()
        res_rgb = res_plotted[:, :, ::-1] # Konversi BGR ke RGB

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)
        with col2:
            st.image(res_rgb, caption="Hasil Deteksi", use_column_width=True)

    # ==========================================
    # LOGIKA PEMROSESAN VIDEO
    # ==========================================
    elif file_extension in ['mp4', 'avi']:
        st.write("Sedang memproses video secara real-time...")
        
        # Streamlit perlu menyimpan video sementara ke server untuk dibaca oleh OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # Buka video menggunakan OpenCV
        vid_cap = cv2.VideoCapture(tfile.name)
        
        # Membuat area kosong di Streamlit untuk diisi frame video terus-menerus
        stframe = st.empty()
        
        while vid_cap.isOpened():
            success, frame = vid_cap.read()
            if success:
                # Prediksi frame dengan YOLOv8
                results = model(frame)
                
                # Gambar Bounding Box
                res_plotted = results[0].plot()
                
                # Konversi format warna untuk Streamlit
                res_rgb = res_plotted[:, :, ::-1]
                
                # Timpa gambar sebelumnya dengan gambar baru (menciptakan efek video live)
                stframe.image(res_rgb, channels="RGB", use_column_width=True)
            else:
                vid_cap.release()
                break
                
        # Menghapus file sementara setelah selesai
        os.remove(tfile.name)
        st.success("Pemrosesan video selesai!")
