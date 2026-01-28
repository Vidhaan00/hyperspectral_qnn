# Copyright (c) 2025 Vidhaan Sinha
# All rights reserved. Proprietary code.
# Author: Vidhaan Sinha
# Patent pending.


import spectral.io.envi as envi
import time
from pre_processing import pre_call
from cnn import train_hsi_cnn
from qnn import train_hsi_qnn
from scipy.io import loadmat

if __name__=="__main__":
    start_time=time.time()
    dark_ref = envi.open(r"/home/vs/HSI/raw_files/0008-1/darkReference.hdr",r"/home/vs/HSI/raw_files/0008-1/darkReference")
    white_ref = envi.open(r"/home/vs/HSI/raw_files/0008-1/whiteReference.hdr", r"/home/vs/HSI/raw_files/0008-1/whiteReference")
    data_ref = envi.open(r"/home/vs/HSI/raw_files/0008-1/raw.hdr",r"/home/vs/HSI/raw_files/0008-1/raw")

    pre_data=pre_call(dark_ref,white_ref,data_ref)
    pre_time=time.time()
    hcube=pre_data["preprocessed_data"].astype(float)
    gtLabel = loadmat(r"/home/vs/HSI/pre_processed_and_gt/gt_0008.mat")["gt_data"] 
    print(f"pre_processing concluded \nruntime:{(pre_time-start_time)/60}min.")
    
    #model, pred_img = train_hsi_cnn(hcube, gtLabel)
    #cnn_time=time.time()
    #print(f"cnn_processing concluded \nruntime:{(cnn_time-pre_time)/60}min.")

    qnn_start_time = time.time()
    q_model, q_pred_img = train_hsi_qnn(hcube, gtLabel)
    qnn_end_time = time.time()
    print(f"QNN processing concluded \nruntime: {(qnn_end_time - qnn_start_time)/60}min.")

