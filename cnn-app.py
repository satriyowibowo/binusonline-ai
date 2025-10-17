import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import time

st.set_page_config(
    page_title="Binus AI - CIFAR-10 Classifier",
    page_icon="🎓",
    layout="wide"
)

st.title("CIFAR-10 CNN Classifier")
st.markdown("**Artificial Intelligence - Image Classification**")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def create_basic_model():
    """Membuat model dasar dengan weights yang lebih baik"""
    from tensorflow.keras import layers
    
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    dummy_input = np.random.rand(1, 32, 32, 3).astype('float32')
    _ = model.predict(dummy_input, verbose=0)
    
    return model

@st.cache_resource
def load_model():
    try:
        model_path = 'models/best_cifar10_model.h5'
        
        if os.path.exists(model_path):
            st.sidebar.info(f"Loading model from: {model_path}")
            
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            if file_size < 1:
                st.sidebar.error(f"Model file too small: {file_size:.2f} MB")
                return create_basic_model()
            
            try:
                model = keras.models.load_model(model_path)
                st.sidebar.success(f"Model loaded! Size: {file_size:.2f} MB")
                return model
            except Exception as load_error:
                st.sidebar.error(f"Error loading: {str(load_error)}")
                return create_basic_model()
                
        else:
            st.sidebar.warning("Custom model not found")
            return create_basic_model()
            
    except Exception as e:
        st.sidebar.error(f"Unexpected error: {str(e)}")
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

model = load_model()

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Image Classification", "Model Information", "About Project"]
)

if app_mode == "Image Classification":
    st.header("Image Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
    with col2:
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption=f"Original: {original_image.size}", use_column_width=True)
            
            if st.button("Classify Image", type="primary"):
                with st.spinner('Processing...'):
                    try:
                        processed_array = preprocess_image(original_image)
                        predictions = model.predict(processed_array, verbose=0)
                        
                        st.sidebar.write("Predictions:", [f"{p:.4f}" for p in predictions[0]])
                        
                        predicted_class_idx = np.argmax(predictions[0])
                        confidence = np.max(predictions[0]) * 100
                        
                        st.success(f"{class_names[predicted_class_idx].title()} ({confidence:.1f}%)")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        y_pos = np.arange(len(class_names))
                        confidences = predictions[0] * 100
                        
                        colors = ['lightcoral' if i != predicted_class_idx else 'limegreen' for i in range(len(class_names))]
                        bars = ax.barh(y_pos, confidences, color=colors, alpha=0.8)
                        
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels([name.title() for name in class_names])
                        ax.invert_yaxis()
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Prediction Confidence')
                        ax.set_xlim(0, 100)
                        
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.2f}%', ha='left', va='center')
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")

elif app_mode == "Model Information":
    st.header("Model Information")

elif app_mode == "About Project":
    st.header("🎓 About This Project")

st.markdown("---")
st.markdown("🎓 **Binus University** - Artificial Intelligence Course")
