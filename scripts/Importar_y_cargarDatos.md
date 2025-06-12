# Bloque 1: Configuración Inicial y Carga de Datos 📂
En esta fase inicial del proyecto, configuramos el entorno de trabajo, importamos las librerías necesarias, definimos rutas y cargamos nuestro conjunto de datos de imágenes médicas. El objetivo es preparar los datos de manera adecuada para las etapas posteriores de preprocesamiento y entrenamiento del modelo.

## 1.1 Importación de Librerías Esenciales
Empezamos importando las librerías fundamentales para el análisis, procesamiento y modelado. En PyTorch, utilizamos torch, torchvision, y además librerías de apoyo como pandas, numpy, os, y scikit-learn. También empleamos imblearn para manejar clases desbalanceadas y matplotlib y seaborn para visualización.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
```
# Configuración para la visualización y uso de GPU si está disponible
```python
sns.set_style("whitegrid")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 1.2 Definición de Rutas y Carga de Datos
Las imágenes están almacenadas en un directorio local, organizado en subcarpetas por categorías (clases). En este ejemplo, las categorías son "Healthy" y "Tumor".

base_path: Directorio raíz que contiene las subcarpetas de categorías.
categories: Lista con los nombres de las subcarpetas, que corresponden a las etiquetas de clase.
Procedemos a recorrer cada carpeta, recolectando las rutas de las imágenes y sus etiquetas, y almacenándolas en un DataFrame para facilitar su manejo.
```python
# Rutas y categorías
base_path = "images"  # Ajusta según tu estructura de directorios
categories = ["Healthy", "Tumor"]  # Nombres de las subcarpetas/clases

# Listas para guardar rutas e labels
image_paths = []
labels = []

# Recolección de datos
for category in categories:
    category_path = os.path.join(base_path, category)
    if os.path.isdir(category_path):
        for image_name in os.listdir(category_path):
            image_paths.append(os.path.join(category_path, image_name))
            labels.append(category)
    else:
        print(f"Advertencia: No se encontró la carpeta '{category}' en '{category_path}'")

# Crear DataFrame con rutas y etiquetas
df = pd.DataFrame({"image_path": image_paths, "label": labels})

# Visualización inicial
print("DataFrame con rutas y etiquetas:")
print(df.head())
print("\nDistribución de clases:")
print(df['label'].value_counts())
```
Al finalizar este bloque, tenemos un DataFrame df que contiene dos columnas principales: image_path con la ruta a cada archivo de imagen, y label con la etiqueta de clase correspondiente (ej. "Healthy" o "Tumor"). Este DataFrame será la base para el preprocesamiento en el siguiente bloque.
