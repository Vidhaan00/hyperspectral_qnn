# Copyright (c) 2025 Vidhaan Sinha
# All rights reserved. Proprietary code. Patent pending.
# Benchmark Pipeline for HELICoiD Brain Cancer Dataset

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import spectral.io.envi as envi

# Import your model training functions
from pre_processing import pre_call
from cnn import train_hsi_cnn
from qnn import train_hsi_qnn

# ---------------------------------------------------------
# 1. UTILITY FUNCTIONS
# ---------------------------------------------------------

def get_balanced_patches(hcube, gt, window_size=25, samples_per_class=2000):
    """Ensures the model doesn't ignore tumor pixels by balancing the training set."""
    height, width, _ = hcube.shape
    half_w = window_size // 2
    X_list, y_list = [], []
    
    unique_classes = np.unique(gt)
    unique_classes = unique_classes[unique_classes != 0] # Skip unlabeled

    for c in unique_classes:
        coords = np.argwhere(gt == c)
        # Keep coords away from borders
        valid_coords = [
            (r, col) for r, col in coords 
            if r >= half_w and r < height - half_w and col >= half_w and col < width - half_w
        ]
        
        n_to_take = min(len(valid_coords), samples_per_class)
        if n_to_take == 0: continue
            
        selected_idx = np.random.choice(len(valid_coords), n_to_take, replace=False)
        for i in selected_idx:
            r, col = valid_coords[i]
            patch = hcube[r-half_w : r+half_w+1, col-half_w : col+half_w+1, :]
            X_list.append(patch)
            y_list.append(c)
            
    return np.array(X_list), np.array(y_list)

def load_patient_data(patient_id):
    """Handles raw ENVI loading and preprocessing for a specific ID."""
    raw_path = f"/home/vs/HSI/raw_files/{patient_id}-1/"
    dark = envi.open(raw_path + "darkReference.hdr", raw_path + "darkReference")
    white = envi.open(raw_path + "whiteReference.hdr", raw_path + "whiteReference")
    raw = envi.open(raw_path + "raw.hdr", raw_path + "raw")
    
    pre_data = pre_call(dark, white, raw)
    hcube = pre_data["preprocessed_data"].astype(float)
    
    # Load Ground Truth if it exists
    gt_path = f"/home/vs/HSI/pre_processed_and_gt/gt_{patient_id}.mat"
    gt = loadmat(gt_path)["gt_data"] if os.path.exists(gt_path) else None
    
    return hcube, gt

# ---------------------------------------------------------
# 2. MAIN BENCHMARK EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    start_bench = time.time()
    
    # Patients with Ground Truth (The Benchmark Core)
    labeled_ids = ["0008", "0012", "0015", "0020"]
    # Patient without Ground Truth (The Blind Clinical Demo)
    blind_id = "0025"
    
    results_summary = []

    # --- PART A: 4-FOLD CROSS VALIDATION ---
    for test_id in labeled_ids:
        print(f"\n{'='*40}")
        print(f"STARTING FOLD: Testing on Patient {test_id}")
        pritn(f"\n{'='*40}")

        # 1. Collect training data from the other 3 labeled patients
        train_ids = [pid for pid in labeled_ids if pid != test_id]
        X_train_total, y_train_total = [], []

        for pid in train_ids:
            h, g = load_patient_data(pid)
            # Use 1500 per class to keep memory/time manageable for QNN
            p, l = get_balanced_patches(h, g, window_size=25, samples_per_class=2000)
            X_train_total.append(p)
            y_train_total.append(l)
            del h, g # Free RAM

        X_train = np.concatenate(X_train_total, axis=0)
        y_train = np.concatenate(y_train_total, axis=0)

        # 2. Load the actual test image for this fold
        test_hcube, test_gt = load_patient_data(test_id)

        # 3. Train Classical CNN
        print(f"Training CNN for Fold {test_id}...")
        cnn_metrics = train_hsi_cnn(X_train, y_train, test_hcube, test_gt,save_dir=f"/home/vs/HSI/results/benchmark/fold_{test_id}/cnn")

        # 4. Train Hybrid QNN
        print(f"Training Hybrid QNN for Fold {test_id}...")
        qnn_metrics = train_hsi_qnn(X_train, y_train, test_hcube, test_gt, 
                                    save_dir=f"/home/vs/HSI/results/benchmark/fold_{test_id}/qnn")
        
        results_summary.append({"fold": test_id, "cnn": cnn_metrics, "qnn": qnn_metrics})

    # --- PART B: BLIND CLINICAL PREDICTION ---
    print(f"\n{'='*40}")
    print(f"FINAL STEP: Blind Prediction on Patient {blind_id}")
    print(f{"\n'='*40}")
    
    blind_hcube, _ = load_patient_data(blind_id)
    # Note: Use a model trained on ALL labeled_ids to predict on blind_id
    # You can call your QNN/CNN with test_gt=None to handle the blind case
    
    end_bench = time.time()
    print(f"\nBenchmark Complete. Total Runtime: {(end_bench - start_bench)/3600:.2f} hours")

    # --- PRINT FINAL TABLE ---
    print("\nFINAL BENCHMARK RESULTS (Kappa Scores)")
    print("Fold | CNN Kappa | QNN Kappa")
    for res in results_summary:
        print(f"{res['fold']} | {res['cnn']['kappa']:.4f} | {res['qnn']['kappa']:.4f}")
