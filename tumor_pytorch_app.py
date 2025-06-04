import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import requests
import tempfile
from PIL import Image
import os
import base64

# Path to your background image
background_image_path = "medical_laboratory.jpg"

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_background(image_path):
    b64_image = get_base64_image(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main-title {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }}
    .prediction-box {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    .stFileUploader > div {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 8px;
    }}
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.95) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background(background_image_path)

# Define your class labels
class_labels = ["Healthy", "Tumor"]

@st.cache(allow_output_mutation=True)
def load_model_from_url(model_url):
    model_path = 'best_model_final.pth'
    # Download the model
    r = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(r.content)
    # Load the model architecture
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: Healthy, Tumor
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model_url = 'https://raw.githubusercontent.com/Sergius-DS/tumor_detect_pytorch/master/best_model_final.pth'

with st.spinner("Loading model... This might take a moment."):
    model = load_model_from_url(model_url)

# Define transformations matching your training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Streamlit UI
st.markdown("""
<div class="main-title">
    <h1>🧠 Deep Learning for Detecting Brain Tumour 🔎</h1>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    predict_button = st.button("Predict")
    # Reset prediction if new file is uploaded
    if uploaded_file:
        if st.session_state.get('uploaded_image') != uploaded_file:
            st.session_state['uploaded_image'] = uploaded_file
            st.session_state['prediction'] = None
    elif st.session_state.get('uploaded_image'):
        uploaded_file = st.session_state['uploaded_image']
    else:
        uploaded_file = None

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', width=240)

# Prediction logic
if predict_button and uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Create mini-batch

    with torch.no_grad():
        output = model(input_batch)
        probs = torch.nn.functional.softmax(output, dim=1)
        probs = probs.numpy()[0]
        predicted_index = np.argmax(probs)
        predicted_class = class_labels[predicted_index]
        confidence = probs[predicted_index]

    # Save prediction to session state
    st.session_state['prediction'] = {
        'class': predicted_class,
        'confidence': confidence
    }

# Display prediction result
if st.session_state.get('prediction'):
    pred = st.session_state['prediction']
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Prediction Result:</h3>
        <p><strong>{pred['class']}</strong> with confidence <strong>{pred['confidence']*100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
