# load_data.py
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pickle
import argparse

# Definir transformaciones
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

# Dataset personalizado
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

def load_and_preprocess(base_path='images', categories=["Healthy", "Tumor"]):
    # Recolectar rutas e etiquetas
    image_paths = []
    labels = []
    for category in categories:
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for image_name in os.listdir(category_path):
                image_paths.append(os.path.join(category_path, image_name))
                labels.append(category)
        else:
            print(f"Advertencia: No se encontró la carpeta '{category}' en '{category_path}'")
    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    print("DataFrame con rutas y etiquetas:")
    print(df.head())
    print("\nDistribución de clases:")
    print(df['label'].value_counts())

    # Codificación de etiquetas
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['label'])

    # División en entrenamiento y conjunto temporal
    X_train_orig, X_temp, y_train_orig, y_temp = train_test_split(
        df[['image_path']], df['category_encoded'], train_size=0.8, shuffle=True, random_state=42, stratify=df['category_encoded']
    )

    # División del conjunto temporal en validación y prueba
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42, stratify=y_temp
    )

    # Sobremuestreo
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_orig, y_train_orig)

    train_df = pd.DataFrame({'image_path': X_train_resampled['image_path'], 'label': y_train_resampled})
    valid_df = pd.DataFrame({'image_path': X_valid['image_path'], 'label': y_valid})
    test_df = pd.DataFrame({'image_path': X_test['image_path'], 'label': y_test})

    # Crear datasets
    train_dataset = BrainTumorDataset(train_df, transform=train_transform)
    valid_dataset = BrainTumorDataset(valid_df, transform=valid_test_transform)
    test_dataset = BrainTumorDataset(test_df, transform=valid_test_transform)

    return train_dataset, valid_dataset, test_dataset, label_encoder

def load_dataset(pkl_path):
    """Carga un dataset serializado desde un archivo pickle."""
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_processed", action='store_true', help="Guardar datasets procesados")
    args = parser.parse_args()

    # Cargar y procesar datos
    train_dataset, valid_dataset, test_dataset, label_encoder = load_and_preprocess()

    # Guardar datasets si se solicita
    if args.save_processed:
        os.makedirs('data', exist_ok=True)
        # Guardar datasets
        with open('data/train_dataset.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open('data/valid_dataset.pkl', 'wb') as f:
            pickle.dump(valid_dataset, f)
        with open('data/test_dataset.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)
        # Guardar label_encoder
        with open('data/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        print("Datasets y label_encoder guardados correctamente.")