## 2.1 Codificación de Etiquetas
En ambos enfoques, es necesario convertir las etiquetas categóricas (por ejemplo, "Healthy" y "Tumor") en valores numéricos. En PyTorch, utilizamos LabelEncoder de scikit-learn para transformar las etiquetas, lo que facilita el manejo en los datasets y modelos.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['label'])
```

Este paso produce una columna con etiquetas numéricas que utilizaremos en los conjuntos de datos.

## 2.2 División de Datos
Es fundamental dividir los datos en conjuntos de entrenamiento, validación y prueba antes de aplicar cualquier técnica de remuestreo o balanceo, para prevenir la fuga de información (data leakage). En PyTorch, esto se realiza con train_test_split, estratificando para mantener la proporción de clases en cada conjunto.

```python
from sklearn.model_selection import train_test_split

# División en entrenamiento (80%) y conjunto temporal (20%)
X_train_orig, X_temp, y_train_orig, y_temp = train_test_split(
    df[['image_path']], df['category_encoded'], train_size=0.8, shuffle=True, random_state=42, stratify=df['category_encoded']
)

# División del conjunto temporal en validación y prueba (50% cada uno)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42, stratify=y_temp
)
```

## 2.3 Sobremuestreo del Conjunto de Entrenamiento
Para abordar el desbalance de clases, usamos RandomOverSampler de imblearn. Solo aplicamos esta técnica en el conjunto de entrenamiento, duplicando aleatoriamente las muestras de la clase minoritaria hasta equilibrar las clases. En PyTorch, esto se refleja en la creación de un DataFrame sobremuestreado.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_orig, y_train_orig)

train_df = pd.DataFrame({'image_path': X_train_resampled['image_path'], 'label': y_train_resampled})
train_df['label'] = train_df['label'].astype(str)
valid_df = pd.DataFrame({'image_path': X_valid['image_path'], 'label': y_valid.astype(str)})
test_df = pd.DataFrame({'image_path': X_test['image_path'], 'label': y_test.astype(str)})
```

Luego, estos DataFrames sirven como base para crear los Dataset y los DataLoader.

## 2.4 Creación de DataFrames para los Generadores en PyTorch
En PyTorch, en lugar de los generadores de Keras, creamos un Dataset personalizado y DataLoaders. Los DataFrames train_df, valid_df y test_df contienen las rutas de las imágenes y las etiquetas, que serán convertidas en tensores en tiempo de carga.

```python
# El DataFrame de entrenamiento sobremuestreado
train_df['label'] = train_df['label'].astype(str)
valid_df['label'] = valid_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)
```

## 2.5 Configuración de Transformaciones de Imagen (Data Augmentation y Preprocesamiento)
En PyTorch, utilizamos transforms.Compose para definir las transformaciones de las imágenes. Para entrenamiento, aplicamos aumentaciones (rotación, flip, etc.), y para validación y prueba solo escalamos y normalizamos.

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

Estos transformadores se usan en la creación de los Dataset.

## 2.6 Creación de Dataset y DataLoader
Con los DataFrames preparados y las transformaciones definidas, creamos los Dataset y los DataLoader en PyTorch para una carga eficiente de las imágenes durante el entrenamiento y evaluación.

```python
from torch.utils.data import Dataset, DataLoader

class BrainTumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = int(self.df.iloc[idx]['label'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Crear los datasets
train_dataset = BrainTumorDataset(train_df, transform=train_transform)
valid_dataset = BrainTumorDataset(valid_df, transform=valid_test_transform)
test_dataset = BrainTumorDataset(test_df, transform=valid_test_transform)

# Crear los DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

## 2.7 Resumen
Con estos pasos, tus datos están listos para ser utilizados en el entrenamiento y evaluación de modelos en PyTorch:

Las etiquetas se han codificado y balanceado mediante sobremuestreo.
Los conjuntos de datos están correctamente divididos y preparados.
Se aplican transformaciones y aumentaciones apropiadas.
Los DataLoaders proporcionan una carga eficiente y en lotes de las imágenes para el entrenamiento y evaluación.
