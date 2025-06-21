# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

# Importar la función build_model desde un archivo separado
from model import build_model
from load_data import load_dataset, BrainTumorDataset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_outputs.extend(outputs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_outputs, all_labels

def train_model(train_loader, valid_loader, device, num_epochs=10, patience=5, initial_lr=1e-3):
    model = build_model(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

    best_auc = 0
    counter = 0

    # Listas para guardar métricas
    train_loss_history = []
    val_loss_history = []
    val_auc_history = []

    train_acc_history = []
    val_acc_history = []

    total_epochs_trained = 0  # contador total de épocas

    # --- Fase 1: Entrenamiento inicial ---
    for epoch in range(num_epochs):
        total_epochs_trained += 1
        print(f"Epoch {total_epochs_trained}/{num_epochs}")

        # Entrenamiento
        train_loss, train_acc, train_outputs, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        # Validación
        model.eval()
        val_loss_total = 0.0
        val_outputs_list = []
        val_labels_list = []
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * images.size(0)

                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                val_outputs_list.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss_total / len(valid_loader.dataset)
        val_acc = correct / total

        # Guardar métricas en las listas
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Cálculo AUC
        val_probs = np.array(val_outputs_list)
        val_labels = np.array(val_labels_list)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.5

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Guardar el mejor modelo
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Mejor modelo guardado")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # --------- Fase 2: Fine-tuning (desbloquear capas) ---------
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Re-optimizar con menor LR
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    fine_tune_epochs = 10

    for epoch in range(fine_tune_epochs):
        total_epochs_trained += 1
        print(f"Fine-tuning Epoch {total_epochs_trained}/{num_epochs + fine_tune_epochs}")

        train_loss, train_acc, train_outputs, train_labels = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validación
        model.eval()
        val_loss_total = 0.0
        val_outputs_list = []
        val_labels_list = []
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * images.size(0)

                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                val_outputs_list.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss_total / len(valid_loader.dataset)
        val_acc = correct / total

        # Guardar métricas de fine-tuning
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Cálculo AUC
        val_probs = np.array(val_outputs_list)
        val_labels = np.array(val_labels_list)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.5
        val_auc_history.append(val_auc)

        print(f"Fine-tuning Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Guardar el mejor modelo
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model_final.pth')
            print("Mejor modelo de fine-tuning guardado")
        else:
            # Opcional: early stopping en fine-tuning
            pass

    # Crear y devolver el diccionario de historial completo
    training_history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'val_auc': val_auc_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'initial_epochs': num_epochs,
        'total_epochs': total_epochs_trained
    }

    return model, training_history

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model", action='store_true', help="Guardar el modelo final")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar datasets serializados
    train_dataset = load_dataset('data/train_dataset.pkl')
    valid_dataset = load_dataset('data/valid_dataset.pkl')

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Entrenar
    model, training_history = train_model(train_loader, valid_loader, device)

    # Guardar modelo final si se solicita
    if args.save_model:
        torch.save(model.state_dict(), 'models/best_model_final.pth')

    # Guardar historia de entrenamiento
    with open('results/training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)

    print("Entrenamiento finalizado y historia guardada.")