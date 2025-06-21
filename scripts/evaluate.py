# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import argparse
from load_data import BrainTumorDataset, valid_test_transform
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from load_data import load_dataset  # Para cargar datasets pickleados

# Importa build_model desde model.py
from model import build_model

def evaluate_model(model, dataloader, label_encoder, device, output_dir):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            probs = outputs.cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    y_probs = np.array(all_probs)
    y_true = np.array(all_labels).astype(int)
    y_pred = (y_probs > 0.5).astype(int)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matriz de Confusión - ResNet en PyTorch')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - ResNet en PyTorch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_training_history(train_acc, val_acc, train_loss, val_loss, initial_epochs, output_path):
    plt.figure(figsize=(12, 6))
    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Exactitud de Entrenamiento')
    plt.plot(val_acc, label='Exactitud de Validación')
    plt.axvline(x=initial_epochs - 1, color='red', linestyle='--', label='Inicio Fine-Tuning')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Pérdida de Entrenamiento')
    plt.plot(val_loss, label='Pérdida de Validación')
    plt.axvline(x=initial_epochs - 1, color='red', linestyle='--', label='Inicio Fine-Tuning')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results", help="Carpeta donde guardar gráficos")
    args = parser.parse_args()

    # Crear la carpeta si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar Dataset pickleado
    dataset_path = 'data/test_dataset.pkl'  # Puedes cambiar a 'valid_dataset.pkl' si prefieres evaluar en validación
    try:
        test_dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error cargando el dataset desde {dataset_path}: {e}")
        exit(1)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Cargar LabelEncoder
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Crear y cargar modelo
    model = build_model(device)
    model.load_state_dict(torch.load('models/best_model_final.pth', map_location=device))
    model.to(device)

    # Evaluar el modelo
    evaluate_model(model, test_loader, label_encoder, device, args.output_dir)

    # Cargar y graficar historia de entrenamiento
    training_history_path = os.path.join('results', 'training_history.pkl')
    if os.path.exists(training_history_path):
        try:
            with open(training_history_path, 'rb') as f:
                training_history_data = pickle.load(f)
            train_acc_history = training_history_data.get('train_acc', [])
            val_acc_history = training_history_data.get('val_acc', [])
            train_loss_history = training_history_data.get('train_loss', [])
            val_loss_history = training_history_data.get('val_loss', [])
            initial_epochs = training_history_data.get('initial_epochs', 10)

            print("\nPlotting training history...")
            plot_training_history(
                train_acc_history,
                val_acc_history,
                train_loss_history,
                val_loss_history,
                initial_epochs,
                os.path.join(args.output_dir, 'training_history_plot.png')
            )
            print("Training history plot saved to", os.path.join(args.output_dir, 'training_history_plot.png'))
        except Exception as e:
            print(f"Error al cargar o graficar historia de entrenamiento: {e}")
    else:
        print(f"No se encontró training_history.pkl en {os.path.join('results')}. Por favor, ejecuta train.py primero.")