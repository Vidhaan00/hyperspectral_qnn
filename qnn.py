# Copyright (c) 2025
# All rights reserved. Proprietary code.
# Author: Vidhaan Sinha
# Hybrid Quantum-Classical Pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import pennylane as qml
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
# Quantum Circuit Definition
# -----------------------------
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# -----------------------------
# Hybrid Model Class
# -----------------------------
class HybridNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, 3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Quantum Layers
        weight_shapes = {"weights": (3, n_qubits)}
        self.qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer3 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer4 = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # Fully Connected
        self.fc1 = nn.Linear(32 * 6 * 6, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 15)
        self.fc4 = nn.Linear(15, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.reshape(-1, 32 * 6 * 6)
        x = F.relu(self.fc1(x))
        
        # Quantum Split & Process
        x_1, x_2, x_3, x_4 = torch.split(x, 5, dim=1)
        x_1 = self.qlayer1(x_1)
        x_2 = self.qlayer2(x_2)
        x_3 = self.qlayer3(x_3)
        x_4 = self.qlayer4(x_4)
        
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# -----------------------------
# Training Entry Point
# -----------------------------
def train_hsi_qnn(hcube, gtLabel, save_dir="/home/vs/HSI/results/0008/qnn_output"):
    _set_seed(0)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Hybrid QNN device:", device)

    height, width, bands = hcube.shape
    
    # PCA & Preprocessing (Same as CNN)
    dim_reduction = 6
    pca = PCA(n_components=dim_reduction)
    imageData = pca.fit_transform(hcube.reshape(-1, bands)).reshape(height, width, dim_reduction)
    imageData = (imageData - np.mean(imageData)) / np.std(imageData)

    # Patch extraction (Same as CNN)
    window_size = 25
    patches, labels = [], []
    for i in range(0, height - window_size + 1):
        for j in range(0, width - window_size + 1):
            patches.append(imageData[i:i+window_size, j:j+window_size, :])
            labels.append(gtLabel[i + window_size // 2, j + window_size // 2])

    patches = np.array(patches)
    labels = np.array(labels).ravel()
    num_classes = len(np.unique(labels))

    # Shuffle for training
    data_list = list(zip(patches, labels))
    random.shuffle(data_list)
    p_shuffled, l_shuffled = zip(*data_list)

    # Dataset Splitting
    split_idx = int(0.75 * len(p_shuffled))
    train_dataset = TensorDataset(torch.from_numpy(np.array(p_shuffled[:split_idx])).permute(0,3,1,2).float(), 
                                  torch.LongTensor(l_shuffled[:split_idx]))
    eval_dataset = TensorDataset(torch.from_numpy(np.array(p_shuffled[split_idx:])).permute(0,3,1,2).float(), 
                                 torch.LongTensor(l_shuffled[split_idx:]))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    model = HybridNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses, val_losses = [], []
    
    # Training Loop
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation Step
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item()
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(v_loss/len(eval_loader))
        print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # Metrics (Identical to CNN)
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('QNN Loss Curves')
    plt.legend()
    plt.savefig(f"{save_dir}/qnn_loss.png")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='jet')
    plt.savefig(f"{save_dir}/qnn_cm.png")

    # Ordered Reconstruction (Fast Batch Method)
    model.eval()
    ordered_preds = []
    # Re-extracting from imageData to ensure spatial order
    all_ordered = []
    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            all_ordered.append(imageData[i:i+window_size, j:j+window_size, :])
    
    recon_tensor = torch.from_numpy(np.array(all_ordered)).permute(0,3,1,2).float()
    for i in range(0, len(recon_tensor), 512):
        batch = recon_tensor[i:i+512].to(device)
        with torch.no_grad():
            out = model(batch)
            ordered_preds.extend(out.argmax(1).cpu().numpy())
    
    resultant_image = np.array(ordered_preds).reshape(height-window_size+1, width-window_size+1)
    plt.figure()
    plt.imshow(resultant_image, cmap='jet')
    plt.savefig(f"{save_dir}/qnn_map.png")

    return model, resultant_image
