import streamlit as st
import cv2
import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
import data_pipeline
import frequency_analysis
import segmentation
import model
from PIL import Image

st.set_page_config(page_title="Low-Frequency Object Detection ", layout="wide")

# Custom CSS for vibrant colorful UI without images
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
        color: #222222;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
        color: #3a0ca3;
        font-weight: 600;
    }
    /* Sidebar header */
    .css-1v3fvcr h2 {
        color: #720026;
        font-weight: 700;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #ff4e50;
        color: white;
        font-weight: 700;
        border-radius: 15px;
        padding: 12px 28px;
        font-size: 16px;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 78, 80, 0.4);
    }
    div.stButton > button:first-child:hover {
        background-color: #e63946;
        color: white;
        box-shadow: 0 6px 8px rgba(230, 57, 70, 0.6);
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #6a0572;
        font-weight: 800;
        text-shadow: 1px 1px 2px #fbc2eb;
    }
    /* Columns background */
    .stColumns > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 12px rgba(255, 78, 80, 0.2);
        margin-bottom: 20px;
    }
    /* File uploader */
    .css-1d391kg .stFileUploader {
        background-color: #ffb5a7;
        border-radius: 15px;
        padding: 15px;
        font-weight: 600;
        color: #6a0572;
    }
    /* Sliders and selectboxes */
    .stSlider > div > div > input[type=range] {
        accent-color: #ff4e50;
    }
    .stSelectbox > div > div > div {
        background-color: #fbc2eb;
        border-radius: 10px;
        color: #6a0572;
        font-weight: 600;
    }
    /* Text inputs */
    .stTextInput > div > input {
        border-radius: 10px;
        border: 2px solid #ff4e50;
        padding: 8px;
        font-weight: 600;
        color: #6a0572;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Low-Frequency Object Detection ")

# Sidebar for navigation
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Single Image Processing", "Batch Processing", "Model Training", "About"]
)

# Add easy mode for doctors
easy_mode = st.sidebar.checkbox("Easy Mode (Doctor Friendly)", value=True)

def get_default_params_for_modality(modality: str):
    if modality == "MRI":
        return {
            "size": 256,
            "filter_type": "gaussian",
            "radius": 30,
            "sigma": 15.0,
        }
    elif modality == "Ultrasound":
        return {
            "size": 256,
            "filter_type": "circular",
            "radius": 20,
            "sigma": 10.0,
        }
    else:
        return {
            "size": 256,
            "filter_type": "circular",
            "radius": 30,
            "sigma": 10.0,
        }

def process_image(img, size, filter_type, radius, sigma):
    preprocessed = data_pipeline.preprocess_image(img, (size, size))
    fshift = frequency_analysis.compute_fft(preprocessed)
    if filter_type == "circular":
        mask = frequency_analysis.create_circular_lowpass_mask(preprocessed.shape, radius)
    else:
        mask = frequency_analysis.create_gaussian_lowpass_mask(preprocessed.shape, sigma)
    filtered_img = frequency_analysis.apply_mask_and_reconstruct(fshift, mask)
    thresh = segmentation.adaptive_threshold(filtered_img)
    cleaned = segmentation.morphological_cleaning(thresh)
    contours = segmentation.find_contours(cleaned)
    mask_img = np.zeros(filtered_img.shape, dtype=np.uint8)
    cv2.drawContours(mask_img, contours, -1, 255, thickness=cv2.FILLED)
    return preprocessed, filtered_img, mask_img

def display_images(original, preprocessed, filtered, mask):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(original, caption="Original Image", use_column_width=True)
    with col2:
        st.image(preprocessed, caption="Preprocessed Image", clamp=True, use_column_width=True)
    with col3:
        st.image(filtered, caption="Frequency Filtered Image", clamp=True, use_column_width=True)
    with col4:
        st.image(mask, caption="Segmented Mask", clamp=True, use_column_width=True)

if app_mode == "Single Image Processing":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if easy_mode:
            modality = st.selectbox("Select Modality", ["MRI", "Ultrasound", "Other"])
            defaults = get_default_params_for_modality(modality)
            size = defaults["size"]
            filter_type = defaults["filter_type"]
            radius = defaults["radius"]
            sigma = defaults["sigma"]
            st.write(f"Using default parameters for {modality}:")
            st.write(f"Resize: {size}, Filter: {filter_type}, Radius: {radius}, Sigma: {sigma}")
        else:
            size = st.slider("Resize dimension", 64, 512, 256, step=32)
            filter_type = st.selectbox("Low-pass filter type", ["circular", "gaussian"])
            radius = st.slider("Radius for circular filter", 1, 100, 30)
            sigma = st.slider("Sigma for gaussian filter", 1.0, 100.0, 10.0)

        if st.button("Process Image"):
            preprocessed, filtered_img, mask_img = process_image(img, size, filter_type, radius, sigma)
            display_images(img, preprocessed, filtered_img, mask_img)

elif app_mode == "Batch Processing":
    input_dir = st.text_input("Input directory path")
    output_dir = st.text_input("Output directory path")
    size = st.slider("Resize dimension", 64, 512, 256, step=32)
    filter_type = st.selectbox("Low-pass filter type", ["circular", "gaussian"])
    radius = st.slider("Radius for circular filter", 1, 100, 30)
    sigma = st.slider("Sigma for gaussian filter", 1.0, 100.0, 10.0)
    if st.button("Process Batch"):
        if not os.path.isdir(input_dir):
            st.error("Input directory does not exist.")
        else:
            os.makedirs(output_dir, exist_ok=True)
            image_paths = glob.glob(os.path.join(input_dir, "*.*"))
            progress_bar = st.progress(0)
            for i, img_path in enumerate(image_paths):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Skip silently without warning
                    continue
                preprocessed, filtered_img, mask_img = process_image(img, size, filter_type, radius, sigma)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_preprocessed.png"), (preprocessed * 255).astype('uint8'))
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_filtered.png"), np.clip(filtered_img, 0, 255).astype('uint8'))
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask_img)
                progress_bar.progress((i + 1) / len(image_paths))
            st.success("Batch processing completed.")

elif app_mode == "Model Training":
    st.write("Train a simple CNN model on segmented patches (placeholder)")
    data_dir = st.text_input("Labeled patches directory")
    epochs = st.number_input("Number of epochs", min_value=1, max_value=100, value=10)
    if st.button("Start Training"):
        st.info("Model training functionality is under development.")

elif app_mode == "About":
    st.markdown("""
    ### Low-Frequency Object Detection Pipeline
    This app provides interactive visualization and batch processing capabilities for detecting low-frequency objects in images.
    
    Features:
    - Single image processing with real-time parameter tuning.
    - Batch processing of image datasets.
    - Placeholder for CNN model training on segmented patches.
    - Modular and extensible design.
    
    Developed using Streamlit, OpenCV, NumPy, and PyTorch.
    """)
