# Python In-built packages
from pathlib import Path
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Kualifikasi Kualitas Cabai | Dashboard",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Kualifikasi Kualitas Cabai")

# Sidebar
st.sidebar.header("Menu Klasifikasi Cabai")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])
    #"Select Task", ['Detection', 'Segmentation'])
 
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Sidebar Pengolahan Citra
st.sidebar.header("Menu Pengolahan Citra")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Histogram', 'Rotasi', 'Closing'])

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            # Otomatis menjalankan fungsi pemrosesan berdasarkan radio button yang dipilih
            if model_type == 'Histogram':
                # Proses histogram
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                
                # Mengambil citra hasil deteksi dalam bentuk NumPy array
                detected_image_np = np.array(res_plotted)

                # Mengubah citra hasil deteksi ke dalam format BGR (Blue, Green, Red)
                detected_image_bgr = cv2.cvtColor(detected_image_np, cv2.COLOR_RGB2BGR)

                # Menghitung histogram citra hasil deteksi
                detected_image_gray = cv2.cvtColor(detected_image_bgr, cv2.COLOR_BGR2GRAY)
                histogram = cv2.calcHist([detected_image_gray], [0], None, [256], [0, 256])

                # Menampilkan histogram di halaman utama (tengah)
                plt.figure()  # Buat objek gambar histogram
                plt.hist(histogram, bins=256, range=(0, 256), density=True, color='r', alpha=0.7)
                st.pyplot(plt)  # Tampilkan gambar histogram di halaman utama (tengah)
                st.title("Histogram Gambar Hasil Deteksi")

            elif model_type == 'Rotasi':
                # Mendefinisikan sudut rotasi (misalnya, 45 derajat)
                angle = 45

                # Melakukan rotasi gambar
                rotated_image = uploaded_image.rotate(angle)

                # Melakukan deteksi pada gambar asli (tanpa rotasi)
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                # Menampilkan gambar hasil deteksi di atas gambar asli
                st.image(res_plotted, caption='Detected Image',
                        use_column_width=True)
                
                # Menampilkan gambar hasil rotasi
                st.image(rotated_image, caption='Rotated Image',
                        use_column_width=True)
                

            elif model_type == 'Closing':
                # Mengkonversi gambar menjadi format OpenCV (BGR)
                image_bgr = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)

                # Melakukan deteksi pada gambar asli
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                # Menampilkan gambar hasil deteksi di atas gambar asli
                st.image(res_plotted, caption='Detected Image',
                        use_column_width=True)

                # Mengkonversi gambar menjadi citra grayscale
                gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

                # Melakukan operasi closing
                kernel = np.ones((5, 5), np.uint8)
                closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

                # Mengkonversi kembali ke citra RGB
                closed_image_rgb = cv2.cvtColor(closed_image, cv2.COLOR_GRAY2RGB)

                # Menampilkan gambar hasil operasi closing
                st.image(closed_image_rgb, caption='Closed Image',
                        use_column_width=True)
                

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

#elif source_radio == settings.WEBCAM:
    #helper.play_webcam(confidence, model)

#elif source_radio == settings.RTSP:
    #helper.play_rtsp_stream(confidence, model)

#elif source_radio == settings.YOUTUBE:
    #helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
