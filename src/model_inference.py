import xarray as xr
import os
import pickle
import numpy as np
import virtual_sensor.Neuron as Neuron
from config_schema import Preprocessing
from scipy.stats import qmc
from model.surrogate_module import SurrogateModule
from model.components import conv1D_surr
import torch
from typing import List

SAVE_PATH = r"C:\Users\romain.ribault\OneDrive - FRANCE ENERGIES MARINES\DIONYSOS\08_DATA\Dataflow"
DATE = "2022-12-01"

envir_dataset_path = os.path.join(SAVE_PATH, DATE.replace('-','\\'), "Environment", "Zefyros", "DataSample_Zefyros.nc")
envir_dataset = xr.open_dataset(envir_dataset_path)

experiment_path = r"C:\Users\romain.ribault\Documents\GitHub\Deep_learning_model_training\outputs\train_surrogate\runs\2023-03-30_08-45-11"

# load preprocessing pipeline
preprocess_path = os.path.join(experiment_path, "preprocessing.pkl")
with open(preprocess_path, 'rb') as f:
    preprocess : Preprocessing = pickle.load(f)

# load model
model_path = os.path.join(experiment_path, r"checkpoints\last.ckpt")
net = conv1D_surr.conv1D_surr(7, 24, 6)
model : SurrogateModule = SurrogateModule.load_from_checkpoint(model_path, net=net)

# Load environment inputs
X_channel_list : List[str] = preprocess.inputs_outputs.envir_variables

# Prepare test data for inference
df= envir_dataset
for dict_key in [{'mag10':'theta10'}, {'hs':'dp'}] : #, {'cur':'cur_dir'}] :
    df = preprocess.feature_eng.get_cos_sin_decomposition(dict_key, df)

X_env = preprocess.split_transform.get_numpy_input_envir_set(df, X_channel_list)
#input_variables_uncertainty = model_object.input_variables_uncertainty

# # Load scalers
# with open(model_object.envir_scaler_path, 'rb') as f:
#     Xscaler = pickle.load(f)
# with open(model_object.spectrum_scaler_path, 'rb') as f:
#     Yscalers = pickle.load(f)
# with open(model_object.pca_path, 'rb') as f:
#     PCAs = pickle.load(f)
x_env = preprocess.envir_scaler.scaler.transform(X_env)
x_env = torch.from_numpy(x_env).float()
# predict nominal spectrum thanks to the surrogate model
model.eval()
y_hat = model(x_env)
y_hat = y_hat.detach().numpy()

# unscale y_hat
Yscalers = preprocess.y_spectrum_scaler.scalers
PCAs = preprocess.decomp_y_spectrum.decomps

# Prepare np arrays to receive inference results
Y_hat_inter = np.zeros_like(y_hat)
batch_size = np.shape(PCAs[0].inverse_transform(y_hat[:,:,0]))[0]
spectrum_length = np.shape(PCAs[0].inverse_transform(y_hat[:,:,0]))[1]
channel_nb = len(Yscalers)
Y_hat = np.zeros((batch_size, spectrum_length, channel_nb ))
i=0
for pca, scaler in zip(PCAs, Yscalers) :
    Y_hat_inter[:,:,i] =  scaler.inverse_transform(y_hat[:,:,i])
    Y_hat[:,:,i] = pca.inverse_transform(Y_hat_inter[:,:,i])
    i+=1

# perform uncertainty propagation and calculate 95% confidence interval
# define input variables uncertainty :
# define variables uncertainty
env_uncertainty = {}
for key in preprocess.inputs_outputs.envir_variables:
    env_uncertainty[key] = 0.2 # +/- 20% uncertainty

sampler = qmc.LatinHypercube(d=len(env_uncertainty), seed=12345)
sample = sampler.random(n=100)

envir_sample = []
y_hat_max_env = []
y_hat_min_env = []

for env_sample in X_env :
    # stretch the sample to the uncertainty range and center it around the nominal value
    for i, key in enumerate(env_uncertainty.keys()):
        sample[:,i] = (sample[:,i] - 0.5 ) * (1 + env_uncertainty[key]) * env_sample[preprocess.inputs_outputs.envir_variables.index(key)]
    # scale the input sample
    x_env_sample = preprocess.envir_scaler.scaler.transform(sample)
    x_env_sample = torch.from_numpy(x_env_sample).float()

    # predict the several spectrum for each sample
    y_sample = model(x_env_sample).detach().numpy()
    
    
    # take the 99% percentice of the sample (none, 512, 6) on axis 1
    y_hat_max_env.append(np.percentile(y_sample, 99.9, axis=0))
    y_hat_min_env.append(np.percentile(y_sample, 0.1, axis=0))
    
y_hat_max_env = np.array(y_hat_max_env)
y_hat_min_env = np.array(y_hat_min_env)

# unscale y_hat
Y_hat_max_env_int= np.zeros_like(y_hat_max_env)
Y_hat_min_env_int= np.zeros_like(y_hat_min_env)
Y_hat_max_env = np.zeros((batch_size, spectrum_length, channel_nb ))
Y_hat_min_env = np.zeros((batch_size, spectrum_length, channel_nb ))
i=0
for pca, scaler in zip(PCAs, Yscalers):
    Y_hat_max_env_int[:,:,i] =  scaler.inverse_transform(y_hat_max_env[:,:,i])
    Y_hat_max_env[:,:,i] =  pca.inverse_transform(Y_hat_max_env_int[:,:,i])
    Y_hat_min_env_int[:,:,i] =  scaler.inverse_transform(y_hat_min_env[:,:,i])
    Y_hat_min_env[:,:,i] = pca.inverse_transform(Y_hat_min_env_int[:,:,i])
    i+=1

# Allocate to Neuron object
Y_channel_list = preprocess.inputs_outputs.neuron_variables
neuron = Neuron.Neuron()
neuron.institution = 'FEM'
neuron.source = 'surrogate_model'

i=0
for channel in Y_channel_list:
    units_dict = preprocess.unit_dictionnary
    units_dict[f'{channel}_max_env'] = units_dict[channel]
    units_dict[f'{channel}_min_env'] = units_dict[channel]
    
    neuron.allocate_ann_psd_inference(channel, units_dict, Y_hat[:,:,i])
    neuron.allocate_ann_psd_inference(f'{channel}_max_env', units_dict, Y_hat_max_env[:,:,i])
    neuron.allocate_ann_psd_inference(f'{channel}_min_env', units_dict, Y_hat_min_env[:,:,i])
    i+=1

#neuron.Frequency_psd = preprocess.Frequency_psd
# Temporary fix for the frequency vector :
neuron.Frequency_psd = preprocess.Frequency_psd.where(preprocess.Frequency_psd>(preprocess.cut_low_frequency), drop=True).values

neuron.time_psd = envir_dataset.time.values
# Save netcdf file
neuron.save_nc(DATE, SAVE_PATH, sensor_name=f'surrogate__Neuron_Tower')

