import streamlit as st
import gdown
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
@st.cache(allow_output_mutation=True)
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
model_file_id = 'YOUR_FILE_ID_HERE'  # Replace with your actual file ID

# Load model
with st.spinner("Loading model from Google Drive..."):
    model = load_pytorch_model_from_gdrive(model_file_id)
  
