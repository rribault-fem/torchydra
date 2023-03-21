
import Env2Mov_Surrogate.Conv1DTranspose_Surrogate as conv1D
import xarray as xr
import os
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from Preprocess_data.Preprocessor_Zefyros_dataset_ANN import get_X_Y_train_test_from_xr_dataset, get_cos_sin_decomposition, Scale_Xscalar_data, Scale_Yspectrum_data, PCA_Yspectrum_data


#### select X and Y variables
ENVIR_VARIABLES = [ 'dp', 'tp', 'hs', 'mag10', 'theta10'] #, 'cur', 'cur_dir']
NEURON_VARIABLES = []
for i in ['x', 'y', 'z']:
    for j in [1,3]:
        NEURON_VARIABLES.append(f'NeurNode{j}AL{i}t_psd')

# select the frequency range to be used for training
cut_low_frequency = 1/35 # 35 is in seconds --> all frequencies below 1/35 are removedurl_base = r"C:\Users\romain.ribault\Documents\Dionysos\echanges_Morphosense\Windeurope_dataset\reference_dataset_Zefyros_windeurope.nc"

# training dataset path : netcdf file containing the training dataset
url_base = r"C:\Users\romain.ribault\Documents\Dionysos\echanges_Morphosense\Windeurope_dataset\reference_dataset_Zefyros_windeurope.nc"

# Set the magnitude associated with direction variables for feature engineering
# The direction variables are replaced by the cos and sin of the angle
# This is done to avoid the discontinuity at 0/360 degrees
envir_direction_dict = {'mag10':'theta10', 'hs':'dp'} #, 'cur':'cur_dir'}

# number of test sample and number of clusters
Test_nb = 40
cluster = 3

unscaled_folder_numpy_data = r'C:\Users\romain.ribault\Documents\GitHub\subsee4d-mooring-monitoring\unscaled_data'
scaled_folder_numpy_data = r'C:\Users\romain.ribault\Documents\GitHub\subsee4d-mooring-monitoring\scaled_data'
scaler_folder = r'C:\Users\romain.ribault\Documents\GitHub\subsee4d-mooring-monitoring\scalers'
pca_folder = r'C:\Users\romain.ribault\Documents\GitHub\subsee4d-mooring-monitoring\pca'
