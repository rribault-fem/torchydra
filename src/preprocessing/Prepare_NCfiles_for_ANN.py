# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 08:55:18 2019

@author: rribault
"""

import xarray as xr
import pandas as pd
import numpy as np
import logging

def MinMaxFinder(X):
    '''
    finds Xmin and Xmax for each channel of the data  
    Parameters
    ----------
    X : np.array - to be scaled data with shape (samples, time_series, Channel_nb)
    
    Returns
    -------
    MinMax :  np.array - each line for each channel [Min; Max]
    '''
    MinMax = np.zeros((len(X[0,0,:]), 2))
    for i in range(len(X[0,0,:])):
        # scale data 0-1 interval    
        xmin = np.min(X[:,:,i])
        xmax = np.max(X[:,:,i])
        MinMax[i,:] = np.array([xmin, xmax])
    return MinMax
    

def scale_function(X , MinMax):
    '''
    Scale X data in [0;1] interval  
    Parameters
    ----------
    X : np.array - to be scaled data with shape (samples, time_series, Channel_nb)
    Channel_nb : number of channel of the input data 
    
    Returns
    -------
    X_scaled :  np.array - scaled data
    Min_Max_matrix : np.array - Min / Max
    '''
    X_scaled = np.zeros(X.shape)
    # for each channel,
    for i in range(len(MinMax)):
        # scale data 0-1 interval    
        xmin = MinMax[i, 0]
        xmax = MinMax[i, 1]
        
        X_train0 = []
        #for each value in the channel
        for j in range (len(X[:,:,i])):
                normalized = (X[j,:,i]-xmin)/(xmax-xmin)
                X_train0.append(normalized)
        X_train0 = np.array(X_train0)
        
        X_scaled[:, :, i] = X_train0
    
    return X_scaled

def unscale_function(X , MinMax):
        '''
        for each channel of X, Unscale from [0,1] interval to real life interval according to MinMax values  
        Parameters
        ----------
        X : np.array - to be unscaled data with shape (samples, time_series, Channel_nb)
        
        Returns
        -------
        X_unscaled :  np.array - unscaled data
        '''
        X_unscaled = np.zeros(X.shape)
        for i in range(len(MinMax)):
            # scale data 0-1 interval    
            xmin = MinMax[i, 0]
            xmax = MinMax[i, 1]
        
            X_Train0 = []
            for j in range (len(X[:,:,i])):
                    unscaled = X[j,:,i] * (xmax-xmin) + xmin
                    X_Train0.append(unscaled)
            X_Train0 = np.array(X_Train0)
            
            X_unscaled[:, :, i] = X_Train0
    
        return X_unscaled

def preprocess_ncdf_to_transcient_ann(df_filename, key) :
    # Load the simulations data
    df = xr.open_dataset(df_filename)
    
    # reshape the data to get the (samples, time, channels) shape for ANN
    
    # get values from Xarray variables
    X = df.X[key].values
    Y = df.Y[key].values
    Z = df.Z[key].values
    Roll = df.Roll[key].values
    Yaw = df.Yaw[key].values
    Pitch = df.Pitch[key].values
    
    
    # Add the channel dimension
    X = np.expand_dims(X, axis = 2)
    Y = np.expand_dims(Y, axis = 2)
    Z = np.expand_dims(Z, axis = 2)
    Roll = np.expand_dims(Roll, axis = 2)
    Yaw = np.expand_dims(Yaw, axis = 2)
    Pitch = np.expand_dims(Pitch, axis = 2)

    #Stack all variables together on the channel axis
    # X_tot = np.append(Yaw, Pitch, axis = 2)
    # X_tot = np.append(X_tot, Roll, axis = 2)
    # X_tot = np.append(X_tot, Z, axis = 2)
    # X_tot = np.append(X_tot, Y, axis = 2)
    # X_tot = np.append(X_tot, X, axis = 2)
    
    # Test only 3 dims - Stack all variables together on the channel axis
    #♀ WARNING : using only X and Y allows a better prediction using real life model
    X_tot = np.append(X, Y, axis = 2)
    X_tot = np.append(X_tot, Yaw, axis = 2)
    
    
    y_tot = df.breakage[key].values
    
    #check same length between X_tot and y_tot
    if len(X_tot) != len(y_tot):
        log = logging.getLogger('train_surrogate') 
        log.info('WARNING: X and Y do not have same length')

    return X_tot, y_tot
    
def preprocess_ncdf_to_tension_prediction(df_filename, key):
    # Load the simulations data
    df = xr.open_dataset(df_filename)
    
    # reshape the data to get the (samples, time, channels) shape for ANN
    
    # get values from Xarray variables
    X       = df.X[key].values
    Y       = df.Y[key].values
    Z       = df.Z[key].values
    Roll    = df.Roll[key].values
    Yaw     = df.Yaw[key].values
    Pitch   = df.Pitch[key].values
    T_M20_1 = df.T_M20_1[key].values
    T_M20   = df.T_M20[key].values
    
    # # context
    # H_dir   = df.H_dir[key].values
    # Hs      = df.Hs[key].values
    # W_mean   = df.W_mean[key].values
    
    # # work H_dir to have same 
    # H_dir_new = np.zeros(X.shape)
    # for i in range(len(H_dir)):
    #     for j in range(X.shape[1]) :
    #         H_dir_new[i:j] = H_dir[i]
            
    # Hs_new = np.zeros(X.shape)
    # for i in range(len(Hs)):
    #     for j in range(X.shape[1]) :
    #         Hs_new[i:j] = Hs[i]
            
    # W_mean_new = np.zeros(X.shape)
    # for i in range(len(W_mean)):
    #     for j in range(X.shape[1]) :
    #         W_mean_new[i:j] = W_mean[i]
    
    
    # Add the channel dimension
    X = np.expand_dims(X, axis = 2)
    Y = np.expand_dims(Y, axis = 2)
    Z = np.expand_dims(Z, axis = 2)
    Roll = np.expand_dims(Roll, axis = 2)
    Yaw = np.expand_dims(Yaw, axis = 2)
    Pitch = np.expand_dims(Pitch, axis = 2)
    
    T_M20_1 = np.expand_dims(T_M20_1, axis = 2)
    T_M20   = np.expand_dims(T_M20, axis = 2)
    
    

    #Stack all variables together on the channel axis
    X_tot = np.append(Yaw, Pitch, axis = 2)
    X_tot = np.append(X_tot, Roll, axis = 2)
    X_tot = np.append(X_tot, Z, axis = 2)
    X_tot = np.append(X_tot, Y, axis = 2)
    X_tot = np.append(X_tot, X, axis = 2)
    
    Y_tot = np.append(T_M20_1, T_M20, axis = 2)
    
    
    #Previous code with all variables stacked together
    
    # stack all the variables on one column
    # X         = np.reshape(X, (X.size, 1))
    # Y         = np.reshape(Y, (Y.size, 1))
    # Z         = np.reshape(Z, (Z.size, 1))
    # Roll      = np.reshape(Roll, (Roll.size, 1))
    # Yaw       = np.reshape(Yaw, (Yaw.size, 1))
    # Pitch     = np.reshape(Pitch, (Pitch.size, 1))
    # T_M20_1   = np.reshape(T_M20_1, (T_M20_1.size, 1))
    # T_M20     = np.reshape(T_M20, (T_M20.size, 1))
    # H_dir_new = np.reshape(H_dir_new, (H_dir_new.size, 1))
    # Hs_new    = np.reshape(Hs_new, (Hs_new.size, 1))
    # W_mean_new    = np.reshape(W_mean_new, (W_mean_new.size, 1))
    
    # # Stack all variables together on the column axis
    # X_tot = np.append(Yaw, Pitch, axis = 1)
    # X_tot = np.append(X_tot, Roll, axis = 1)
    # X_tot = np.append(X_tot, Z, axis = 1)
    # X_tot = np.append(X_tot, Y, axis = 1)
    # X_tot = np.append(X_tot, X, axis = 1)
    # X_tot = np.append(X_tot, H_dir_new, axis = 1)
    # X_tot = np.append(X_tot, Hs_new, axis = 1)
    # X_tot = np.append(X_tot, W_mean_new, axis = 1)
    
    #y_tot = np.append(T_M20_1, T_M20, axis = 1)
    
    return X_tot, Y_tot

def preprocess_ncdf_to_fatigue_regressor(df_filename, key) :
    # Load the simulations data
    df = xr.open_dataset(df_filename)
    
    # reshape the data to get the (samples, time, channels) shape for ANN
    
    # get values from Xarray variables
    X = df.X[key].values
    Y = df.Y[key].values
    Z = df.Z[key].values
    Roll = df.Roll[key].values
    Yaw = df.Yaw[key].values
    Pitch = df.Pitch[key].values
    Fatigue_damage = df.Fatigue_damage[key].values
    
    
    # Add the channel dimension
    X = np.expand_dims(X, axis = 2)
    Y = np.expand_dims(Y, axis = 2)
    Z = np.expand_dims(Z, axis = 2)
    Roll = np.expand_dims(Roll, axis = 2)
    Yaw = np.expand_dims(Yaw, axis = 2)
    Pitch = np.expand_dims(Pitch, axis = 2)

    # Stack all variables together on the channel axis
    X_tot = np.append(Yaw, Pitch, axis = 2)
    X_tot = np.append(X_tot, Roll, axis = 2)
    X_tot = np.append(X_tot, Z, axis = 2)
    X_tot = np.append(X_tot, Y, axis = 2)
    X_tot = np.append(X_tot, X, axis = 2)
    
    # y_tot = envir.breakage[key].values
    
    y_tot = Fatigue_damage
    
    #check same length between X_tot and y_tot
    if len(X_tot) != len(y_tot):
        log = logging.getLogger('train_surrogate') 
        log.info('WARNING: X and Y do not have same length')

    return X_tot, y_tot


    