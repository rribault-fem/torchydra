# -*- coding: utf-8 -*-
"""
Created on 06/12/2022

@author: rribault
"""

import xarray as xr
import numpy as np
import pywt
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from sklearn.decomposition import PCA
import logging
import math
 
# file = r'X:\MONAMOOR\Zefyros_simulations_netcdf_V3_22-11-25.nc'
# df = xr.open_dataset(file)

def get_numpy_input_envir_set(df, envir_variables) :
    # loading channels data in numpy for CNN 
    
    input_envir_set = np.empty_like(np.expand_dims(df[envir_variables[0]].values, axis = 1))
    for env_var in envir_variables :
        input_envir = np.expand_dims(df[env_var].values, axis = 1)
        input_envir_set = np.append(input_envir_set, input_envir, axis = 1)
    input_envir_set = np.delete(input_envir_set, 0, axis=1)
    return input_envir_set

def get_numpy_input_channel_set(df, channels) :
    # loading channels data in numpy for CNN 

    input_channel_set = np.empty_like(np.expand_dims(df[channels[0]].values, axis = 2))
    for channel in channels :
        input_channel = np.expand_dims(df[channel].values, axis = 2)
        input_channel_set = np.append(input_channel_set, input_channel, axis = 2)
    input_channel_set = np.delete(input_channel_set, 0, axis=2)
    return input_channel_set

#range(1,128)


###
### find allowable training & test set
###

def get_valid_training_samples_for_one_test_sample_on_one_variable(df_training, df_test, test_time_index, variable, envir_bin) :
    # catch key error on df_test : 
    try : 
        lte = df_test[variable].sel(time=test_time_index).values + envir_bin[variable]
        gte = df_test[variable].sel(time=test_time_index).values - envir_bin[variable]
    except KeyError :
        log = logging.getLogger('train_surrogate')
        log.info('KeyError : variable {} not found in df_test'.format(variable))
        return False
        
    lte = float(lte)
    gte = float(gte)
    
    df_valid = df_training.isel(time = (df_training[variable]>gte) & (df_training[variable]<lte))
    
    return df_valid

def find_training_and_test_set(df, test_set_number, cluster_nb, envir_bin)  :
    count = 0
    found=False
    max_guesses_allowed = 200
    log = logging.getLogger('train_surrogate')
    log.info('#####')
    log.info("start guessing valid training / test set with the following environmental bin :")
    log.info(str(envir_bin))
    log.info(f'looking for {test_set_number} test samples in training set, divided in {cluster_nb} clusters ')
    
    while not found and count < max_guesses_allowed:

        training_list = np.arange(0,len(df.time.values))

        div = int(len(df.time.values) / cluster_nb)

        test_list = []
        for i in range(cluster_nb) :
            if i == cluster_nb :
                test_list += random.sample(range(i*div, len(training_list)), int(test_set_number/cluster_nb)) 
            else: test_list += random.sample(range(i*div, (i+1)*div), int(test_set_number/cluster_nb))

        test_list.sort()

        training_list=np.delete(training_list, test_list)

        df_test = df.isel(time=test_list)
        df_training = df.isel(time=training_list)

        nb_training_sample_in_bin = []


        for test_time in df_test.time.values :

            df_valid = df_training
            list_var=''
            for envir_var in envir_bin.keys() :
                df_valid = get_valid_training_samples_for_one_test_sample_on_one_variable(df_valid, df_test, test_time, envir_var, envir_bin)
                list_var = list_var + ' & ' + envir_var
                # log = logging.getLogger('train_surrogate') log.info('Nb samples with valid {} : {} '.format(envir_var, str(len(df_valid.time)) ))
            nb_training_sample_in_bin.append(len(df_valid.time))

        nb_training_sample_in_bin_dict = {
            "attrs":{
                "description" : "int representing the number of valid training sample for the current test set"},
            "dims" : "time",
            "data" : nb_training_sample_in_bin,
            "name" : "nb_training"
        } 
        DataArray_nb_training  =  xr.DataArray.from_dict(nb_training_sample_in_bin_dict)
        Dataset_nb_training = xr.Dataset(
            { "nb_training" : DataArray_nb_training},
            coords={"time" : df_test.time.values}
        )
        df_test = xr.merge([df_test, Dataset_nb_training], combine_attrs="drop_conflicts")

        if float(df_test.nb_training.min().values) >= 1 :
            found = True
            
            log.info(' Success : test_set found. returning valid dataset with len {} vs. training set '.format(str(len(df_test.time))))
            log.info('#####')
            return df_training, df_test
        elif count%10 == 0 :
            log.info(f'all guess until {count} failed. Keep trying')
        count +=1

    return False, False


def compute_wavelet_coeff_for_CNN(input_set, wavelet_scales,  waveletname) :
    log = logging.getLogger('train_surrogate')
    log.info('####')
    log.info('start computing wavelet')
    log.info('####')
    input_data_cwt = np.ndarray(shape=(np.shape(input_set)[0], len(wavelet_scales), len(wavelet_scales),np.shape(input_set)[2] ))

    for ii in range(0,np.shape(input_set)[0]):
        if ii % 10 == 0:
            log.info('computing for sample nb {}'.format(ii))
        for jj in range(0,np.shape(input_set)[2]):
            signal = input_set[ii, :, jj]
            coeff, freq = pywt.cwt(signal, wavelet_scales, waveletname, 1)
            coeff_ = coeff[:,:len(wavelet_scales)]
            input_data_cwt[ii, :, :, jj] = coeff_

    return input_data_cwt

# scale data
def MinMax_scale_4d_matrix(tdata, mmx_list=None) :
    tdata_transformed = np.zeros_like(tdata)
    n_samples = np.shape(tdata_transformed)[0]
    n_wavelets_scales = np.shape(tdata_transformed)[1]
    n_channels = np.shape(tdata_transformed)[3]


    channel_scalers = []

    for i in range(n_channels):
        slc = tdata[:, :, :, i].reshape(n_samples, n_wavelets_scales**2) # make it a bunch of row vectors
        if mmx_list is None :
            mmx = MinMaxScaler()
            transformed = mmx.fit_transform(slc)
        else :
            mmx = mmx_list[i]
            transformed = mmx.transform(slc)
        transformed = transformed.reshape(n_samples, n_wavelets_scales, n_wavelets_scales) # reshape it back to tiles
        tdata_transformed[:, :, :, i] = transformed # put it in the transformed array
        channel_scalers.append(mmx) # store the transform
    
    return channel_scalers, tdata_transformed

# Apply PCA on each channel for spectrum
def PCA_3d_matrix(tdata,n_component, pca_list=None) :
    
    n_samples = np.shape(tdata)[0]
    n_freq = np.shape(tdata)[1]
    n_channels = np.shape(tdata)[2]
    tdata_transformed = np.zeros((n_samples, n_component, n_channels))


    channel_pcas = []

    for i in range(n_channels):
        slc = tdata[:, :, i] # make it a bunch of row vectors
        if pca_list is None :
            pca = PCA(n_components=n_component)
            transformed = pca.fit_transform(slc)
        else :
            pca = pca_list[i]
            transformed = pca.transform(slc)
        tdata_transformed[:, :, i] = transformed # put it in the transformed array
        channel_pcas.append(pca) # store the transform
    
    return channel_pcas, tdata_transformed


# scale data
def MinMax_scale_3d_matrix(tdata, mmx_list=None) :
    tdata_transformed = np.zeros_like(tdata)
    n_samples = np.shape(tdata_transformed)[0]
    n_freq = np.shape(tdata_transformed)[1]
    n_channels = np.shape(tdata_transformed)[2]


    channel_scalers = []

    for i in range(n_channels):
        slc = tdata[:, :, i] # make it a bunch of row vectors
        if mmx_list is None :
            mmx = MinMaxScaler()
            transformed = mmx.fit_transform(slc)
        else :
            mmx = mmx_list[i]
            transformed = mmx.transform(slc)
        tdata_transformed[:, :, i] = transformed # put it in the transformed array
        channel_scalers.append(mmx) # store the transform
    
    return channel_scalers, tdata_transformed

def get_anomaly_list_from_df(df) :
    anomaly_catalogue = {
    'GBoxEff' : 1,
    }

    anomaly_list = []
    count=0
    log = logging.getLogger('train_surrogate')
    for i in range(len(df.time)):
        flag = df.Flag_Anomaly.values[i]
        for key in anomaly_catalogue.keys():
            if key in str(flag) : 
                anomaly_list.append(anomaly_catalogue[key])
                found_flag = True
            count +=1

            log.info('count: '+ str(count))
            log.info('keys len :'+ str(len(anomaly_catalogue.keys())))
            if (not found_flag) :
                anomaly_list.append(0)

    return anomaly_list

def get_xr_dataset_time(array_name : str, array_values : np.array, description : str , time_values : np.array) :
    
    data_array_dict = {
            "attrs":{
                "description" : description},
            "dims" : "time",
            "data" : array_values,
            "name" : array_name
        }

    DataArray  =  xr.DataArray.from_dict(data_array_dict)
    Dataset = xr.Dataset(
        { array_name : DataArray},
        coords={"time" : time_values}
    )

    return Dataset

def get_X_Y_train_test_from_xr_dataset(df, envir_bin, X_channel_list = ['tp', 'hs', 'mag10', 'theta10'], Y_channel_list = ['X_psd', 'Y_psd'], df_train_set_envir = None, test_nb=40, clusters = 3, cut_low_frequency=1/35) :

    # First split training and test data with customs requirement on the validity domain of a model.
    df_training_set, df_test_set = find_training_and_test_set(df, test_nb, clusters, envir_bin)

    # anomaly = False
    # if 'Flag_anomaly' in X_channel_list :
    #     anomaly = True
    #     # Prepare Y anomaly
    #     anomaly_list = get_anomaly_list_from_df(df)
    #     description =  "annomaly catalogue number for each time"
    #     array_name = 'anomaly'
    #     Dataset_annomaly_catalogue_nb = get_xr_dataset_time(array_name, anomaly_list, description, df.time.values)
    #     df = xr.merge([df, Dataset_annomaly_catalogue_nb], combine_attrs="drop_conflicts")
    #     # y_training_anomaly_set = df['anomaly_catalogue_nb'].sel(time=df_training_set.time.values)
    #     # y_test_anomaly_set = df['anomaly_catalogue_nb'].sel(time=df_test_set.time.values)
    #     # Y_channel_list.drop('anomaly')

    #Prepare Y
    log = logging.getLogger('train_surrogate')
    Y_numpy_channels_training_set = get_numpy_input_channel_set(df_training_set, Y_channel_list)
    log.info('# selected input channels are : {}'.format(str(Y_channel_list)))
    log.info('# channel training set numpy transformation success. Shape of channel training is {} '.format(str(np.shape(Y_numpy_channels_training_set))))
    Y_numpy_channels_test_set = get_numpy_input_channel_set(df_test_set, Y_channel_list)
    log.info('# channel test set numpy transformation success. Shape of channel test is {}'.format(np.shape(Y_numpy_channels_test_set)))

    # Cut off frequency to not try to predict noise
    cut_low_freq_arg = np.argwhere(df.Frequency_psd.values>(cut_low_frequency))[0][0]
    #cut_high_freq = np.argwhere(df.Frequency_psd.values<4)[0][0]
    if math.log(cut_low_freq_arg,2) - int(math.log(cut_low_freq_arg,2)) != 0 :
        cut_low_freq_arg = 512
    Y_numpy_channels_training_set = Y_numpy_channels_training_set[:,cut_low_freq_arg:,:]
    Y_numpy_channels_test_set = Y_numpy_channels_test_set[:, cut_low_freq_arg :  ,:]

        
    #envir_variables = ['cur_dir', 'cur', 'dp', 'tp', 'hs', 'mag10', 'theta10', 'gust10']

    X_numpy_channels_training_set = get_numpy_input_envir_set(df_training_set, X_channel_list)
    X_numpy_channels_test_set = get_numpy_input_envir_set(df_test_set, X_channel_list)

    # Save training set with only environmental values

    drop_vars = list(df_training_set.keys())
    vars_to_remove = [var for var in drop_vars if var not in X_channel_list]
    df_training_set = df_training_set.drop(vars_to_remove)
    os.makedirs(os.path.dirname(df_train_set_envir), exist_ok=True)
    df_training_set.to_netcdf(os.path.join(df_train_set_envir))

 
    return X_numpy_channels_training_set, X_numpy_channels_test_set, Y_numpy_channels_training_set, Y_numpy_channels_test_set, cut_low_freq_arg

def Scale_Xscalar_data(X_numpy_channels_training_set, X_numpy_channels_test_set, file_names= ['x_train_scaled', 'x_val_scaled'], folder = None) :
    log = logging.getLogger('train_surrogate')
    log.info('###')
    log.info('scale the data')
    
    scaler = MinMaxScaler()
    scaler.fit(X_numpy_channels_training_set)
    x_numpy_channels_training_set = scaler.transform(X_numpy_channels_training_set)
    x_numpy_channels_val_set = scaler.transform(X_numpy_channels_test_set)
    

    log.info('X  data scaled')
    log.info('###')

    return x_numpy_channels_training_set, x_numpy_channels_val_set, scaler
    
def Scale_Yspectrum_data(Y_numpy_channels_training_set:np.array, Y_numpy_channels_test_set:np.array, file_names = ['y_train_scaled', 'y_val_scaled'], folder = None) -> np.array:    
        
    Y_channel_scalers, y_numpy_channels_training_set = MinMax_scale_3d_matrix(Y_numpy_channels_training_set)
    Y_channel_scalers, y_numpy_channels_val_set = MinMax_scale_3d_matrix(Y_numpy_channels_test_set, Y_channel_scalers)

    return y_numpy_channels_training_set, y_numpy_channels_val_set, Y_channel_scalers

    # return X_scaled_numpy_channels_training_set, X_scaled_numpy_channels_val_set, Y_scaled_numpy_channels_training_set, Y_scaled_numpy_channels_val_set

    # l1 = [np.around(a,6) for a in np.arange(16.5, 200, 0.5)][::-1] # Low frequencies (example : slow drift)
    # l2 = [np.around(a,6) for a in np.arange(2, 16, 0.1)][::-1] # Wave frequencies including rotor frequencies
    # l3 = [np.around(1/i,6) for i in np.arange(0.52,4.01,0.1)] # High Frequencies  (0.52 not to have duplicate 0.5 and 4.01 to include 4.0)
    # wavelet_scales =l1 + l2 + l3
    # waveletname = 'morl'

    # numpy_channels_cwt_training_set = compute_wavelet_coeff_for_CNN(numpy_channels_training_set, wavelet_scales,  waveletname)
    # numpy_channels_cwt_test_set = compute_wavelet_coeff_for_CNN(numpy_channels_test_set, wavelet_scales,  waveletname)

    # log = logging.getLogger('train_surrogate') log.info('###')
    # log = logging.getLogger('train_surrogate') log.info('scale the data')
    # channel_scalers, np_scaled_channels_cwt_tr_set = MinMax_scale_4d_matrix(numpy_channels_cwt_training_set)
    # channel_scalers, np_scaled_channels_cwt_ts_set = MinMax_scale_4d_matrix(numpy_channels_cwt_test_set, channel_scalers)

# a function which decompose 0 to 365 values to cos and sin values
def get_cos_sin_decomposition(dict_key:dict, df:xr.Dataset):
    """
    This function decomposes 0 to 365 values into cosine and sine values. It takes in a dictionary and an xarray Dataset as input.

    The function first logs information about the decomposition. 
    It then calculates the cosine and sine values for each magnitude and angle specified in the input dictionary.

    The calculated cosine and sine values are then merged into the input Dataset.

    Args:
        dict_key (dict): A dictionary containing magnitude-angle pairs to be decomposed.
        df (xr.Dataset): An xarray Dataset containing the data to be decomposed.

    Returns:
        xr.Dataset: An xarray Dataset containing the original data as well as the calculated cosine and sine values.
    """
    log = logging.getLogger('train_surrogate')
    log.info('###')
    log.info(f'get cos and sin decomposition of the data {dict_key}')
    for magnitude, angle in dict_key.items() :
        df_cos = get_xr_dataset_time(magnitude+'_cos', df[magnitude]*np.cos(2*np.pi*df[angle].values/365), f'{magnitude} * cos of {angle} values', df.time.values)
        df_sin = get_xr_dataset_time(magnitude+'_sin', df[magnitude]*np.sin(2*np.pi*df[angle].values/365), f'{magnitude} * sin of {angle} values', df.time.values)

    df = xr.merge([df, df_cos, df_sin], compat = 'no_conflicts')

    return df

def PCA_Yspectrum_data(Y_numpy_channels_training_set:np.array, Y_numpy_channels_test_set:np.array, file_names = ['Y_train_pca', 'Y_val_pca'], folder = None, n_component=24) -> np.array:    
    log = logging.getLogger('train_surrogate')
    log.info('###')
    log.info('PCA the data')
    Y_channel_pcas, Y_numpy_channels_training_set = PCA_3d_matrix(Y_numpy_channels_training_set, n_component)
    Y_channel_pcas, Y_numpy_channels_val_set = PCA_3d_matrix(Y_numpy_channels_test_set, n_component, pca_list=Y_channel_pcas)
    return Y_numpy_channels_training_set, Y_numpy_channels_val_set, Y_channel_pcas