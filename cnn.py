# Copyright (c) 2025
# All rights reserved. Proprietary code.
# Author: Vidhaan Sinha
# Patent pending.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


# -----------------------------
# Reproducibility
# -----------------------------
def _set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# CNN training entry point
# -----------------------------
def train_hsi_cnn(hcube, gtLabel, save_dir="/home/vs/HSI/results/0008/cnn_output"):

    _set_seed(0)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CNN device:", device)

    height, width, bands = hcube.shape

    # -----------------------------
    # PCA (fit once)
    # -----------------------------
    dim_reduction = 6
    pca = PCA(n_components=dim_reduction)
    pixels = hcube.reshape(-1, bands)
    pixels_pca = pca.fit_transform(pixels)
    imageData = pixels_pca.reshape(height, width, dim_reduction)

    # Normalize
    imageData = (imageData - np.mean(imageData)) / np.std(imageData)

    # -----------------------------
    # Patch extraction
    # -----------------------------
    window_size = 25
    patches, labels = [], []

    for i in range(0, height - window_size + 1):
        for j in range(0, width - window_size + 1):
            patches.append(imageData[i:i+window_size, j:j+window_size, :])
            labels.append(gtLabel[i + window_size // 2,
                                  j + window_size // 2])

    patches = np.array(patches)
    labels = np.array(labels).ravel()

    # Shuffle
    data = list(zip(patches, labels))
    random.shuffle(data)
    patches, labels = zip(*data)
    patches, labels = np.array(patches), np.array(labels)

    # -----------------------------
    # Train / validation split
    # -----------------------------
    split_idx = int(0.75 * len(patches))
    train_patches, eval_patches = patches[:split_idx], patches[split_idx:]
    train_labels, eval_labels = labels[:split_idx], labels[split_idx:]

    train_dataset = TensorDataset(
        torch.Tensor(train_patches).permute(0,3,1,2),
        torch.LongTensor(train_labels)
    )
    eval_dataset = TensorDataset(
        torch.Tensor(eval_patches).permute(0,3,1,2),
        torch.LongTensor(eval_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    num_classes = len(np.unique(labels))
    print("Number of classes:", num_classes)

    # -----------------------------
    # CNN (IDENTICAL TO CODE 1)
    # -----------------------------
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(6, 16, 3, stride=1, padding=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.fc1 = nn.Linear(32 * 6 * 6, 20)
            self.fc2 = nn.Linear(20, 20)
            self.fc3 = nn.Linear(20, 15)
            self.fc4 = nn.Linear(15, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.reshape(-1, 32 * 6 * 6)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)

    model = CNN().to(device)

    # -----------------------------
    # Training setup
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 20

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                preds = out.argmax(1)

                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_losses.append(val_loss / len(eval_loader))
        val_accs.append(100 * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Acc: {train_accs[-1]:.2f}% | "
              f"Val Acc: {val_accs[-1]:.2f}%")

    # -----------------------------
    # Learning curve
    # -----------------------------
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cnn_loss_curve.png", dpi=1000)
    plt.close()

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="jet")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cnn_confusion_matrix.png", dpi=1000)
    plt.close()

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

#    model.eval()
#    batch_size = 256
#    predictions = []
#
#    with torch.no_grad():
#        for i in range(0, len(patches), batch_size):
#            batch = torch.Tensor(
#                patches[i:i+batch_size]
#            ).permute(0,3,1,2).to(device)
#
#            outputs = model(batch)
#            preds = outputs.argmax(dim=1)
#            predictions.extend(preds.cpu().numpy())
#
#            del batch, outputs, preds
#            torch.cuda.empty_cache()
#
#    expected_shape = (
#        height - window_size + 1,
#        width - window_size + 1
#    )
#    resultant_image = np.array(predictions).reshape(expected_shape)


    # Predict and display results using batch processing
    # --- FIX: Generate a clean, ordered set of patches for reconstruction ---
    # --- OPTIMIZED FIX: Ordered batch processing for speed ---
    model.eval()
    predictions = []
    
    # 1. Create the ordered tensor on CPU
    # We use the original imageData which hasn't been shuffled
    all_ordered_patches = []
    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            all_ordered_patches.append(imageData[i:i+window_size, j:j+window_size, :])
    
    # Convert to tensor (keeping it on CPU to save VRAM)
    ordered_patches_tensor = torch.tensor(np.array(all_ordered_patches)).permute(0, 3, 1, 2).float()
    
    # 2. Process in batches for speed
    batch_size = 512 # Larger batch size is fine for inference
    with torch.no_grad():
        for i in range(0, len(ordered_patches_tensor), batch_size):
            batch = ordered_patches_tensor[i : i + batch_size].to(device)
            outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            
            del batch, outputs # Keep GPU memory clean

    # 3. Reshape and Save
    expected_shape = (height - window_size + 1, width - window_size + 1)
    resultant_image = np.array(predictions).reshape(expected_shape)
    plt.figure(figsize=(6,6))
    plt.imshow(resultant_image, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cnn_prediction_map.png", dpi=1000)
    plt.close()

    return model, resultant_image
