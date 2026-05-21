import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import zipfile
import io
import os
from collections import defaultdict

# ==========================================
# Streamlit Config
# ==========================================

st.set_page_config(
    page_title="Military Aircraft Sorter",
    layout="wide"
)

st.title("Military Aircraft")

st.write(
    """
    Foto hochladen
    """
)

# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ==========================================
# Upload Multiple Images
# ==========================================

uploaded_files = st.file_uploader(
    "Bilder hochladen",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ==========================================
# Process Images
# ==========================================

if uploaded_files:

    st.success(f"✅ {len(uploaded_files)} Bilder hochgeladen")

    categorized_images = defaultdict(list)

    progress = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")

        # YOLO Prediction
        results = model.predict(image, conf=0.4)

        result = results[0]

        # ======================================
        # Klassen erkennen
        # ======================================

        detected_classes = []

        if result.boxes is not None:

            for cls_id in result.boxes.cls.tolist():

                class_name = model.names[int(cls_id)]

                if class_name not in detected_classes:
                    detected_classes.append(class_name)

        # Falls nichts erkannt
        if not detected_classes:
            detected_classes = ["Unbekannt"]

        # Bild in Kategorien speichern
        for category in detected_classes:

            categorized_images[category].append({
                "filename": uploaded_file.name,
                "image": image
            })

        progress.progress((idx + 1) / len(uploaded_files))

    st.success("✅ Sortierung abgeschlossen!")

    # ==========================================
    # Kategorien anzeigen
    # ==========================================

    st.header("📂 Sortierte Kategorien")

    for category, images in categorized_images.items():

        st.subheader(f"📁 {category} ({len(images)} Bilder)")

        cols = st.columns(4)

        # ZIP-Datei vorbereiten
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            for i, item in enumerate(images):

                image = item["image"]
                filename = item["filename"]

                # Bild anzeigen
                with cols[i % 4]:
                    st.image(
                        image,
                        caption=filename,
                        use_container_width=True
                    )

                # Bild temporär speichern
                temp_buffer = io.BytesIO()
                image.save(temp_buffer, format="PNG")

                zip_file.writestr(
                    filename,
                    temp_buffer.getvalue()
                )

        zip_buffer.seek(0)

        # Download Button für Kategorie
        st.download_button(
            label=f"⬇️ Kategorie '{category}' herunterladen",
            data=zip_buffer,
            file_name=f"{category}.zip",
            mime="application/zip"
        )

        st.divider()
