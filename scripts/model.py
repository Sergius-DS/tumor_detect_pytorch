# model.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def build_model(device):
    # Load ResNet50 with the default, most up-to-date ImageNet weights
    # This addresses the UserWarning about 'pretrained' and 'weights'.
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Congelar todas las capas si quieres
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar la capa final
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    model = model.to(device)
    return model