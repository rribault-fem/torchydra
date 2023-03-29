import tensorflow as tf
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from preprocessing.Preprocessor_Zefyros_dataset_ANN import get_valid_training_samples_for_one_test_sample_on_one_variable
# from torch.utils.tensorboard import SummaryWriter
from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle

@dataclass
class SurrogateModel(ABC):

    df_training_path: str
    cut_low_frequency: int
    input_variables: List[str]
    output_variables: List[str]
    modelname: str
    units_dictionnary : dict

    @property
    def Frequency_psd(self):
        df = xr.open_dataset(self.df_training_path)
        return df['Frequency_psd'].where(df['Frequency_psd']>(self.cut_low_frequency), drop=True)

    def set_input_variables(self, input_variables : list):
        self.input_variables = input_variables

    def set_output_variables(self, output_variables : list):
        self.output_variables = output_variables
    
    def set_modelname(self, name : str, model_version: str = 'v0'):
        self.modelname = name+'_'+model_version

    
    def save_model(self):
            # save model
            file = f'{self.modelname}.pickle'
            with open(file, 'wb') as f:
                    pickle.dump(self, f)

    
    @abstractmethod
    def train_surrogate(self, x_train : np.array, x_test: np.array, y_train: np.array, y_test: np.array,
                         epochs : int =100, batch_size : int =32, loss : str = 'mse', learn_rate : float =0.001, dropout_rate : float =0.3, verbose : int =1) :
        """Train the autoencoder on the given data"""

        
    def get_nb_training_sample_in_validity_domain(self, df_envir : xr.Dataset, test_time : int):
        """Check if the given environmental variables are in the validity domain of the surrogate model"""

        """ _summary_

        _extended_summary_

        test_time : the time for to check the validity domain. used in a df_envir.sel(time=test_time)
        df_envir : xarray Dataset with environmental values. It shall have the same environmental variable as en environmental_bin

        :return: the number of training sample in the validity domain
        :rtype: int
        """

        df_training = xr.open_dataset(self.df_training_path)
        df_valid = df_training
        for envir_var in self.environmental_bin.keys() :
            df_valid = get_valid_training_samples_for_one_test_sample_on_one_variable(df_valid, df_envir, test_time, envir_var, self.environmental_bin)
    
        return int(len(df_valid.time))