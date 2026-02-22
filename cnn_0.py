# Copyright (c) 2025 Vidhaan Sinha
# All rights reserved. Proprietary code. Patent pending.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, classification_report
from scipy.ndimage import median_filter # For map smoothing

def _set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Enhanced feature extraction to handle patient variability
        self.conv1 = nn.Conv2d(6, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout prevents the model from "memorizing" one patient
        self.dropout = nn.Dropout(0.3)
        
        # Flattened size for 25x25 input after two 2x2 pools is 6x6
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.reshape(-1, 64 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def train_hsi_cnn(X_train, y_train, test_hcube, test_gt, save_dir,test_id):
    _set_seed(0)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Identify classes
    num_classes = int(np.max(y_train) + 1)
    
    # 1. Prepare Data
    train_ds = TensorDataset(
        torch.tensor(X_train).permute(0, 3, 1, 2).float(),
        torch.tensor(y_train).long()
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = CNN(num_classes).to(device)
    
    # 2. STRATEGIC WEIGHTING
    # We punish the model more for missing Tumor (Class 2) and Blood (Class 1)
    # Weights: [Background, Blood, Tumor, Healthy]
    # Adjust these based on your specific integer labels
    weights = torch.tensor([0.1, 10.0, 10.0, 0.4]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 3. Training Loop
    num_epochs = 30 # Increased epochs for better convergence
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # 4. Inference and Spatial Smoothing
    model.eval()
    window_size = 25
    h, w, c = test_hcube.shape
    
    # Extract and predict
    ordered_patches = []
    for i in range(h - window_size + 1):
        for j in range(w - window_size + 1):
            ordered_patches.append(test_hcube[i : i+window_size, j : j+window_size, :])
    
    ordered_tensor = torch.tensor(np.array(ordered_patches)).permute(0, 3, 1, 2).float()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(ordered_tensor), 512):
            batch = ordered_tensor[i : i + 512].to(device)
            out = model(batch)
            predictions.extend(out.argmax(1).cpu().numpy())

    result_raw = np.array(predictions).reshape(h - window_size + 1, w - window_size + 1)
    
    # 5. APPLY MEDIAN FILTER (To get the "First Upload" look)
    # This removes isolated misclassified pixels (noise)
    resultant_image = median_filter(result_raw, size=7)

    # 6. Save and Report
    plt.figure(figsize=(10, 10))
    plt.imshow(resultant_image, cmap='jet')
    plt.axis('off')
    plt.savefig(f"{save_dir}/cnn_map_final.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    metrics = {"kappa": 0.0}
    if test_gt is not None:
        pred_h, pred_w = resultant_image.shape
        gt_cropped = test_gt[window_size//2 : window_size//2 + pred_h, 
                             window_size//2 : window_size//2 + pred_w]
        
        y_true = gt_cropped.flatten()
        y_pred = resultant_image.flatten()
        
        # We explicitly define the class labels we expect from HELICoiD
        # 0: Background, 1: Normal, 2: Tumor, 3: Blood/Vessel
        target_names = ['Background', 'Normal', 'Tumor', 'Vessels']
        all_labels = [0, 1, 2, 3]

        kappa = cohen_kappa_score(y_true, y_pred)
        
        # 'labels' ensures that classes missing in the test patient are shown as 0.0
        report = classification_report(y_true, y_pred, 
                                       labels=all_labels, 
                                       target_names=target_names, 
                                       zero_division=0)
        
        print(f"\n--- Fold Metrics for Patient {test_id} ---")
        print(f"Kappa Score: {kappa:.4f}")
        print(report)
        
        with open(f"{save_dir}/cnn_metrics.txt", "w") as f:
            f.write(f"Kappa: {kappa}\n")
            f.write(report)
        
        metrics["kappa"] = kappa

    return metrics
