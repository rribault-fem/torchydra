import tensorflow as tf
import xarray as xr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv1DTranspose, Dropout, Add, Activation, BatchNormalization, Conv1D, RepeatVector 
from tensorflow.keras import backend as K

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing.Preprocessor_Zefyros_dataset_ANN import get_valid_training_samples_for_one_test_sample_on_one_variable

class Surrogate(tf.keras.Model) :

    def __init__(self):
        super().__init__()

    def set_scalers_paths(self, envir_scaler : str, spectrum_scaler : str):
        self.envir_scaler_path = envir_scaler
        self.spectrum_scaler_path = spectrum_scaler

    def set_pca_path(self, pca_path : str):
        self.pca_path = pca_path

    def set_training_set(self, df_training_path : str):
        self.df_training_path = df_training_path

    def set_Frequency_psd(self, cut_low_frequency : int):
        "set_Frequency_psd sets the frequency psd to be used for the training"
        df = xr.open_dataset(self.df_training_path)
        self.Frequency_psd = df['Frequency_psd'].where(df['Frequency_psd']>(cut_low_frequency), drop=True)

    def set_input_variables(self, input_variables : list):
        self.input_variables = input_variables

    def set_output_variables(self, output_variables : list):
        self.output_variables = output_variables
    
    def set_name(self, name : str):
        self.name = name

    def set_units_dictionnary(self, units_dictionnary : dict):
        """set_units_dictionnary sets the inputs / outputs units dictionnary

        :param units_dictionnary: _description_
        :type units_dictionnary: dict
        """
        self.units_dictionnary = units_dictionnary
    

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
