import xarray as xr
import os
import pickle
import numpy as np
import virtual_sensor.export_data.Neuron as Neuron
from config_schema import Preprocessing
from scipy.stats import qmc
from model.surrogate_module import SurrogateModule
import torch
from typing import List
import yaml
from utils.load_env_file import load_env_file
import hydra

load_env_file("env.yaml")

SAVE_PATH = os.environ["INFER_SAVE_PATH"]
DATE = "2022-12-01"

envir_dataset_path = os.path.join(SAVE_PATH, DATE.replace('-','\\'), "Environment", "Zefyros", "DataSample_Zefyros.nc")
envir_dataset = xr.open_dataset(envir_dataset_path)


experiment_path = os.environ["INFER_EXPERIMENT_PATH"]

# load preprocessing pipeline
# Use of pickle object to load the scaler already fit to the data
preprocess_path = os.path.join(experiment_path, "preprocessing.pkl")
with open(preprocess_path, 'rb') as f:
    preprocess : Preprocessing = pickle.load(f)

# load model
model_path = os.path.join(experiment_path, r"checkpoints\last.ckpt")
hydra_config_path = os.path.join(experiment_path, r'.hydra/config.yaml' )

with open(hydra_config_path, 'r') as f:
    hydra_config = yaml.safe_load(f)

# load a dummy model with dummy kwargs to load the checkpoint
kwargs = {'x_input_size': 7, 'spectrum_decomp_length': 512, 'spectrum_channel_nb': 18}
net: torch.nn.Module = hydra.utils.instantiate(hydra_config['model_net'], **kwargs)

# model kwargs parameters are infered from checkpoint
model : SurrogateModule = SurrogateModule.load_from_checkpoint(model_path, net=net)

# Load environment inputs
X_channel_list : List[str] = preprocess.inputs_outputs.envir_variables

# Prepare test data for inference
df= envir_dataset
envir_direction_dict = preprocess.feature_eng.envir_direction_dict
for dict_key in envir_direction_dict :
    df = preprocess.feature_eng.get_cos_sin_decomposition(dict_key, df)
X_env = preprocess.split_transform.get_numpy_input_envir_set(df, X_channel_list)
x_env = preprocess.envir_scaler.scaler.transform(X_env)

# predict nominal spectrum thanks to the surrogate model
def model_predict(x_env: np.array, model :SurrogateModule ) -> np.ndarray :
    
    x_env = torch.from_numpy(x_env).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_env = x_env.to(device)
    model = model.to(device)
    model.eval()
    y_hat = model(x_env)
    y_hat = y_hat.detach().cpu().numpy()
    
    return y_hat

y_hat = model_predict(x_env, model)

# unscale y_hat
Yscalers = preprocess.y_spectrum_scaler.scalers

if preprocess.perform_decomp :
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

else : 
    # Prepare np arrays to receive inference results
    Y_hat = np.zeros_like(y_hat)
    batch_size = np.shape(Yscalers[0].inverse_transform(y_hat[:,:,0]))[0]
    spectrum_length = np.shape(Yscalers[0].inverse_transform(y_hat[:,:,0]))[1]
    channel_nb = len(Yscalers)
    Y_hat = np.zeros((batch_size, spectrum_length, channel_nb ))


    i=0
    for scaler in  Yscalers :
        Y_hat[:,:,i] =  scaler.inverse_transform(y_hat[:,:,i])
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
    
    y_sample = model_predict(x_env, model)
    

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

if preprocess.perform_decomp :
    i=0
    for pca, scaler in zip(PCAs, Yscalers):
        Y_hat_max_env_int[:,:,i] =  scaler.inverse_transform(y_hat_max_env[:,:,i])
        Y_hat_max_env[:,:,i] =  pca.inverse_transform(Y_hat_max_env_int[:,:,i])
        Y_hat_min_env_int[:,:,i] =  scaler.inverse_transform(y_hat_min_env[:,:,i])
        Y_hat_min_env[:,:,i] = pca.inverse_transform(Y_hat_min_env_int[:,:,i])
        i+=1
else :
    i=0
    for scaler in Yscalers:
        Y_hat_max_env[:,:,i] =  scaler.inverse_transform(y_hat_max_env[:,:,i])
        Y_hat_min_env[:,:,i] =  scaler.inverse_transform(y_hat_min_env[:,:,i])
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
neuron.Frequency_psd = preprocess.Frequency_psd.where(preprocess.Frequency_psd>(preprocess.split_transform.cut_low_frequency), drop=True).values

neuron.time_psd = envir_dataset.time.values
# Save netcdf file
neuron.save_nc(DATE, SAVE_PATH, ann_name=f'surrogate_Neuron_Tower')


