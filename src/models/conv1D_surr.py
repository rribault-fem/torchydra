import tensorflow as tf
import xarray as xr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv1DTranspose, Dropout, Add, Activation, BatchNormalization, Conv1D, RepeatVector 
from tensorflow.keras import backend as K
from surrogate import Surrogate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing.Preprocessor_Zefyros_dataset_ANN import get_valid_training_samples_for_one_test_sample_on_one_variable

class conv1D_surrogate(Surrogate) :

    def __init__(self, spectrum_channel_nb:int):
        super().__init__()
        self.Dense1 = Dense(128, activation='relu')
        self.Dense2 = Dense(64, activation='relu')
        self.Dense3 = Dense(32, activation='relu')
        self.Dense4 = Dense(16, activation='relu')

        # Reshape encoded input to a 3D tensor with shape (None, 1, 16)
        self.Reshape((16, 1))

        # Decode the encoded input using Conv1DTranspose
        # first use wide kernel size to learn the global structure and interraction between channels
        self.Conv1DT1(32, kernel_size=32, strides=2, padding='same', input_shape=(16, 1), activation='relu')
        self.Conv1DT2(64, kernel_size=16, strides=2, padding='same',activation='relu')
        self.Conv1DT3(128, kernel_size=8, strides=2, padding='same',activation='relu')
        # then use narrow kernel size to learn the local structure
        self.Conv1DT4(64, kernel_size=4, strides=2, padding='same',activation='relu')
        self.Conv1DT5(32, kernel_size=2, strides=2, padding='same',activation='relu')
        # finally filter the output to get the desired output shape
        self.Conv1D(spectrum_channel_nb, kernel_size=4, strides=2, padding='same',activation='relu')
        self.Conv1D2(spectrum_channel_nb, kernel_size=4, strides=2, padding='same',activation='relu')
        self.Conv1D3(spectrum_channel_nb, kernel_size=4, strides=2, padding='same',activation='relu')
        self.Conv1D4(spectrum_channel_nb, kernel_size=4, strides=2, padding='same',activation='relu')

    def call(self, inputs):
        # Encode the input using dense layer
        x = self.Dense1(inputs)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)

        # Reshape encoded input to a 3D tensor with shape (None, 1, 16)
        x = self.Reshape(x)

        # Decode the encoded input using Conv1DTranspose
        # first use wide kernel size to learn the global structure and interraction between channels
        x = self.Conv1DT1(x)
        x = self.Conv1DT2(x)
        x = self.Conv1DT3(x)
        # then use narrow kernel size to learn the local structure
        x = self.Conv1DT4(x)
        x = self.Conv1DT5(x)
        # finally filter the output to get the desired output shape
        for i in range (0, 4):
            x = self.Conv1D(x)
        return x