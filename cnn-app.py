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

st.title(" CIFAR-10 CNN Classifier")
st.markdown("**Artificial Intelligence - Image Classification**")

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
        model_path = 'models/best_cifar10_model.h5'
        
        if not os.path.exists(model_path):
            st.sidebar.warning("Model file not found. Using basic model.")
            return create_basic_model()
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        if file_size < 1:
            st.sidebar.warning(f"⚠️ Model file seems too small ({file_size:.2f} MB). Using basic model.")
            return create_basic_model()
        
        st.sidebar.info(f"🔄 Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        st.sidebar.success(f"✅ Custom model loaded successfully! Size: {file_size:.2f} MB")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading custom model: {str(e)}")
        st.sidebar.info("🔄 Falling back to basic model...")
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
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        st.subheader("Preprocessing Options")
        show_processed = st.checkbox("Show processed image (32x32)", value=True)
        
        st.sidebar.subheader("Model Status")
        model_path = 'models/best_cifar10_model.h5'
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            st.sidebar.success(f"Custom model loaded")
            st.sidebar.info(f"Size: {file_size:.2f} MB")
        else:
            st.sidebar.warning("Using basic model")
            st.sidebar.info("Custom model file not found")
        
    with col2:
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(original_image, caption=f"Original Size: {original_image.size}", use_column_width=True)
            
            if show_processed:
                processed_image = original_image.resize((32, 32))
                st.image(processed_image, caption="Processed (32x32)", use_column_width=True)
            
            st.subheader("Prediction")
            
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
                        
                        result_col1, result_col2, result_col3 = st.columns(3)
                        
                        with result_col1:
                            st.metric(
                                label="Predicted Class",
                                value=class_names[predicted_class_idx].title(),
                                delta=f"{confidence:.1f}%"
                            )
                        
                        with result_col2:
                            st.metric(
                                label="Class Index",
                                value=predicted_class_idx
                            )
                            
                        with result_col3:
                            st.metric(
                                label="Confidence",
                                value=f"{confidence:.2f}%"
                            )
                        
                        st.sidebar.subheader("Prediction Debug")
                        st.sidebar.write(f"Raw scores: {[f'{p*100:.2f}%' for p in predictions[0]]}")
                        
                        st.subheader("Confidence Distribution")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        y_pos = np.arange(len(class_names))
                        confidences = predictions[0] * 100
                        
                        colors = ['lightcoral' if i != predicted_class_idx else 'limegreen' for i in range(len(class_names))]
                        bars = ax.barh(y_pos, confidences, color=colors, alpha=0.8)
                        
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels([f"{name.title()} ({i})" for i, name in enumerate(class_names)])
                        ax.invert_yaxis()
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Prediction Confidence for Each Class')
                        ax.set_xlim(0, 100)
                        
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.2f}%', ha='left', va='center', fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                        
                        st.pyplot(fig)
                        
                        st.subheader("📋 Detailed Prediction Scores")
                        
                        scores_data = []
                        for i, (class_name, score) in enumerate(zip(class_names, predictions[0])):
                            scores_data.append({
                                'Class Name': class_name.title(),
                                'Class Index': i,
                                'Confidence Score': f"{score*100:.4f}%",
                                'Raw Probability': f"{score:.6f}",
                                'Is Predicted': '✅' if i == predicted_class_idx else ''
                            })
                        
                        st.dataframe(scores_data, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.info("Please try with a different image or check the model file.")
        
        else:
            st.info("Please upload an image to start classification")
            st.markdown("""
            Supported categories: Airplane, Automobile, Bird, Cat, DeerDog, Frog, Horse, Ship, Truck
            """)

elif app_mode == "Model Information":
    st.header("Model Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("CNN Architecture")
        st.markdown("""
        """)
        
        st.subheader("Training Details")
        st.markdown("""
        """)
    
    with col2:
        st.subheader("Model Summary")
        if model is not None:
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()
            st.text_area("Model Architecture", summary_string, height=400)
        
        st.subheader("Model Status")
        model_path = 'models/best_cifar10_model.h5'
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            st.success("Custom model file found")
            st.write(f"**File location:** `{model_path}`")
            st.write(f"**File size:** {file_size:.2f} MB")
            st.write(f"**Last modified:** {time.ctime(os.path.getmtime(model_path))}")
        else:
            st.warning("Using basic model")
            st.info("Custom model file not found. The app is using a basic CNN model with random weights.")
            st.write("**Expected path:** `models/best_cifar10_model.h5`")

elif app_mode == "About Project":
    st.header("About This Project")
    
    st.markdown("""
 
    """)
    
    st.subheader("CIFAR-10 Dataset Classes")
    cols = st.columns(5)
    
    for i, class_name in enumerate(class_names):
        with cols[i % 5]:
            st.markdown(f"**{class_name.title()}**")
            st.write(f"Class {i}")

st.markdown("---")
st.markdown(
    "<p>Artificial Intelligence Course</p>",
    unsafe_allow_html=True
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
