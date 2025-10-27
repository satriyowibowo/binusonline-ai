import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import time

MODEL_FILE_NAME = "best_model_lr_0_0001.h5"
MODEL_PATH = f"models/{MODEL_FILE_NAME}"

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="",
    layout="wide"
)

try:
    query_params = st.query_params
except AttributeError:
    query_params = st.experimental_get_query_params()

if 'health_check' in query_params:
    st.write("OK")
    st.stop()

st.title("CIFAR-10 CNN Classifier")
st.markdown("**Artificial Intelligence for Image Classification**")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def create_basic_model():
    """Membuat model dasar jika model utama tidak tersedia"""
    from tensorflow.keras import layers
    
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.sidebar.warning(f"Model file not found: {MODEL_PATH}")
            return create_basic_model()
        
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if file_size < 1:
            st.sidebar.warning(f"Model file seems too small ({file_size:.2f} MB)")
            return create_basic_model()
        
        st.sidebar.info(f"Loading model: {MODEL_FILE_NAME}")
        model = keras.models.load_model(MODEL_PATH)
        st.sidebar.success(f"Model loaded! Size: {file_size:.2f} MB")
        return model
        
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return create_basic_model()

def preprocess_image(img):
    """Preprocess gambar untuk model CIFAR-10"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.sidebar.title("Navigation")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Image Classification", "Model Information", "About Project"]
)

model = load_model()

if app_mode == "Image Classification":
    st.header("Image Classification")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        st.subheader("Image Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption=f"Original Image: {original_image.size}", use_column_width=True)
        
        with col2:
            processed_image = original_image.resize((32, 32))
            st.image(processed_image, caption="Processed Image (32x32)", use_column_width=True)
        
        if st.button("Classify Image", type="primary", use_container_width=True):
            with st.spinner('Processing image and making prediction...'):
                try:
                    processed_array = preprocess_image(original_image)
                    start_time = time.time()
                    predictions = model.predict(processed_array, verbose=0)
                    prediction_time = time.time() - start_time
                    
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100
                    
                    st.success(f"Prediction Completed in {prediction_time:.2f}s!")
                    
                    st.subheader("Prediction Results")
                    pred_col1, pred_col2 = st.columns(2)
                    
                    with pred_col1:
                        st.subheader("Prediction Metrics")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                label="Predicted Class",
                                value=class_names[predicted_class_idx].title()
                            )
                        
                        with metric_col2:
                            st.metric(
                                label="Class Index",
                                value=predicted_class_idx
                            )
                            
                        with metric_col3:
                            st.metric(
                                label="Confidence",
                                value=f"{confidence:.2f}%"
                            )
                        
                        st.subheader("Confidence Distribution")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        y_pos = np.arange(len(class_names))
                        confidences = predictions[0] * 100
                        
                        colors = ['lightcoral' if i != predicted_class_idx else 'limegreen' for i in range(len(class_names))]
                        bars = ax.barh(y_pos, confidences, color=colors, alpha=0.8)
                        
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels([f"{name.title()}" for name in class_names])
                        ax.invert_yaxis()
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Confidence for Each Class')
                        ax.set_xlim(0, 100)
                        
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.2f}%', ha='left', va='center', fontweight='bold')
                        
                        st.pyplot(fig)
                    
                    with pred_col2:
                        st.subheader("Detailed Prediction Scores")
                        
                        scores_data = []
                        for i, (class_name, score) in enumerate(zip(class_names, predictions[0])):
                            scores_data.append({
                                'Class Name': class_name.title(),
                                'Class Index': i,
                                'Confidence Score': f"{score*100:.4f}%",
                                'Raw Probability': f"{score:.6f}",
                                'Is Predicted': 'Yes' if i == predicted_class_idx else 'No'
                            })
                        
                        st.dataframe(scores_data, use_container_width=True, height=400)
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    else:
        st.info("Please upload an image to start classification")
        st.markdown("""
        **Tips for best results:**
        - Use clear, well-lit images
        - Center the main object in the image
        - Avoid blurry or distorted images
        - Supported categories: 
          - Airplane, Automobile, Bird, Cat, Deer
          - Dog, Frog, Horse, Ship, Truck
        """)

elif app_mode == "Model Information":
    st.header("Model Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("CNN Architecture")
        st.markdown("""
        **Model Structure:**
        - Input Layer: 32x32x3 (RGB images)
        - Data Augmentation: Random Flip, Rotation, Zoom
        - Conv Block 1: Conv2D (32 filters) + BatchNorm + MaxPooling
        - Conv Block 2: Conv2D (64 filters) + BatchNorm + MaxPooling  
        - Conv Block 3: Conv2D (128 filters) + BatchNorm + MaxPooling
        - Global Average Pooling
        - Dropout: 0.4
        - Output Layer: Dense (10 units, softmax)
        """)
        
        st.subheader("Training Details")
        st.markdown("""
        - Dataset: CIFAR-10
        - Total Classes: 10
        - Image Size: 32x32 pixels
        """)
    
    with col2:
        st.subheader("Model Summary")
        if model is not None:
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()
            st.text_area("Model Architecture", summary_string, height=400)
        
        st.subheader("Model Status")
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.success("Custom model file found")
            st.write(f"**File:** `{MODEL_FILE_NAME}`")
            st.write(f"**Size:** {file_size:.2f} MB")
            st.write(f"**Path:** `{MODEL_PATH}`")
            st.write(f"**Last modified:** {time.ctime(os.path.getmtime(MODEL_PATH))}")
        else:
            st.warning("Using basic model")
            st.info(f"Custom model file not found: {MODEL_PATH}")

elif app_mode == "About Project":
    st.header("About This Project")
    
    st.markdown("""    
    **Project**: CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)
    
    ### Project Overview
    This application demonstrates the implementation of a CNN model for classifying images 
    into 10 categories from the CIFAR-10 dataset.
    
    ### Technical Stack
    - Deep Learning: TensorFlow 2.x, Keras
    - Web Framework: Streamlit
    - Containerization: Docker
    - Deployment: NAS
    """)
    
    st.subheader("CIFAR-10 Dataset Classes")
    cols = st.columns(5)
    
    for i, class_name in enumerate(class_names):
        with cols[i % 5]:
            st.markdown(f"**{class_name.title()}**")
            st.write(f"Class {i}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Artificial Intelligence Course</p>"
    "<p>CIFAR-10 CNN Classification | Built with Streamlit & Docker</p>"
    "</div>",
    unsafe_allow_html=True
)