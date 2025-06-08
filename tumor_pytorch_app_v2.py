import streamlit as st
import gdown
import base64
import os
import tempfile
import torch
from torchvision import models, transforms
from PIL import Image

# Path to your background image
background_image_path = "medical_laboratory.jpg"

# Function to set background (unchanged)
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
    /* ... other styles ... */
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background(background_image_path)

# Function to download and load PyTorch model from Google Drive
@st.cache_data
def load_pytorch_model_from_gdrive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        gdown.download(url, tmp_file.name, quiet=False)
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, 1),
            torch.nn.Sigmoid()
        )
        model.load_state_dict(torch.load(tmp_file.name, map_location=torch.device('cpu')))
        os.unlink(tmp_file.name)
    model.eval()
    return model

# Your Google Drive file ID for the model
model_file_id = '1DbD-05UT7n68K1lPHgVw82HVN5dleezg'  # Replace with your actual file ID

# Load model
with st.spinner("Loading model from Google Drive..."):
    model = load_pytorch_model_from_gdrive(model_file_id)

# Transformaciones iguales a las de entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Interfaz de usuario
st.markdown("""
<div class="main-title">
    <h1>🧠 Detección de Tumor Cerebral con Deep Learning 🔎</h1>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    predict_button = st.button("Predecir")
    # Mantener la imagen cargada si ya se cargó
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
        st.image(image, caption='Imagen cargada.', width=240)

# Predicción
if predict_button and uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Mini-batch

    with torch.no_grad():
        output = model(input_batch)
        prob = output.item()  # valor entre 0 y 1
        if prob > 0.5:
            predicted_class = "Tumor"
            confidence = prob
        else:
            predicted_class = "Healthy"
            confidence = 1 - prob

    # Guardar en estado de sesión
    st.session_state['prediction'] = {
        'class': predicted_class,
        'confidence': confidence
    }

# Mostrar resultado
if st.session_state.get('prediction'):
    pred = st.session_state['prediction']
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Resultado de la Predicción:</h3>
        <p><strong>{pred['class']}</strong> con confianza <strong>{pred['confidence']*100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
  
