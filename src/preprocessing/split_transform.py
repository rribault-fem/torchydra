import logging
from dataclasses import dataclass
import os
import numpy as np
import xarray as xr
from typing import List, Optional, Tuple
import random

@dataclass
class Split_transform :
    test_nb : int
    cluster: int
    split_method : str
    envir_bin : dict

    def process_data(self, df : xr.Dataset, X_channel_list : List[str], Y_channel_list : List[str], df_train_set_envir_filename: str ) -> Tuple[np.array, np.array]:
        log = logging.getLogger('train_surrogate')
        log.info('###')

        # Select the split method defined in hydra config file.
        # The user can define a custom split method by adding a method to this class and then replace the split_method name to the hydra config file
        if hasattr(self, self.split_method):
            log.info(f"Splitting data into training and test sets with method <{self.split_method}>")
            df_training_set, df_test_set = getattr(self, self.split_method)(df)
        else : raise ValueError(f"No such method in {__name__}: {self.split_method}")

        # Save the training set environment inside experiment folder
        self.save_training_environment(df_training_set, X_channel_list, df_train_set_envir_filename)
        log.info('data splitted, now transform to np.array')

        # Transform the data to numpy array
        X_train, X_test, Y_train, Y_test = self.transform_train_test_xr_dataset_to_numpy(df_training_set, df_test_set,  X_channel_list = X_channel_list, Y_channel_list = Y_channel_list)
        log.info('data transformed to np.array')
        log.info('###')

        return  X_train, X_test, Y_train, Y_test
    
    def transform_train_test_xr_dataset_to_numpy(self, 
                                                 df_training_set :xr.Dataset, 
                                                 df_test_set: xr.Dataset,
                                                 X_channel_list:List[str], 
                                                 Y_channel_list = List[str], 
                                                 df_train_set_envir = None, 
                                                 cut_low_frequency=1/35) :


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
        Y_numpy_channels_training_set = self.get_numpy_input_channel_set(df_training_set, Y_channel_list)
        log.info('# selected input channels are : {}'.format(str(Y_channel_list)))
        log.info('# channel training set numpy transformation success. Shape of channel training is {} '.format(str(np.shape(Y_numpy_channels_training_set))))
        Y_numpy_channels_test_set = self.get_numpy_input_channel_set(df_test_set, Y_channel_list)
        log.info('# channel test set numpy transformation success. Shape of channel test is {}'.format(np.shape(Y_numpy_channels_test_set)))

        # Cut off frequency to not try to predict noise
        cut_low_freq_arg = np.argwhere(df_training_set.Frequency_psd.values>(cut_low_frequency))[0][0]
        #cut_high_freq = np.argwhere(df.Frequency_psd.values<4)[0][0]
        Y_numpy_channels_training_set = Y_numpy_channels_training_set[:,cut_low_freq_arg:,:]
        Y_numpy_channels_test_set = Y_numpy_channels_test_set[:, cut_low_freq_arg :  ,:]


        X_numpy_channels_training_set = self.get_numpy_input_envir_set(df_training_set, X_channel_list)
        X_numpy_channels_test_set = self.get_numpy_input_envir_set(df_test_set, X_channel_list)
    
        return X_numpy_channels_training_set, X_numpy_channels_test_set, Y_numpy_channels_training_set, Y_numpy_channels_test_set
    
    def save_training_environment(self, df_training_set : xr.Dataset, X_channel_list : List[str], df_train_set_envir_filename:str):
        
        # Save training set with only environmental values
        drop_vars = list(df_training_set.keys())
        vars_to_remove = [var for var in drop_vars if var not in X_channel_list]
        df_training_set = df_training_set.drop(vars_to_remove)
        os.makedirs(os.path.dirname(df_train_set_envir_filename), exist_ok=True)
        df_training_set.to_netcdf(os.path.join(df_train_set_envir_filename))

    def find_test_set_in_model_validity_domain(self, df )  :
        """This function is used to find a valid test set.
        The validity domain of the model is defined by the environmental bin.
        A test sample is within validity domain of the model if it exist a training sample "close" to the test sample.
        "Close" means that all the environmental variables of the test sample are within the user defined environmental bin.
        """
        
        count = 0
        found=False
        max_guesses_allowed = 200
        log = logging.getLogger('train_surrogate')
        log.info('#####')
        log.info("start guessing valid training / test set with the following environmental bin :")
        log.info(str(self.envir_bin))
        log.info(f'looking for {self.test_nb} test samples in training set, divided in {self.cluster} clusters ')
        
        while not found and count < max_guesses_allowed:

            training_list = np.arange(0,len(df.time.values))

            div = int(len(df.time.values) / self.cluster)

            test_list = []
            for i in range(self.cluster) :
                if i == self.cluster :
                    test_list += random.sample(range(i*div, len(training_list)), int(self.test_nb/self.cluster)) 
                else: test_list += random.sample(range(i*div, (i+1)*div), int(self.test_nb/self.cluster))

            test_list.sort()

            training_list=np.delete(training_list, test_list)

            df_test = df.isel(time=test_list)
            df_training = df.isel(time=training_list)

            nb_training_sample_in_bin = []


            for test_time in df_test.time.values :

                df_valid = df_training
                list_var=''
                for envir_var in self.envir_bin.keys() :
                    df_valid = self.get_valid_training_samples_for_one_test_sample_on_one_variable(df_valid, df_test, test_time, envir_var, self.envir_bin)
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

    def get_valid_training_samples_for_one_test_sample_on_one_variable(self, df_training, df_test, test_time_index, variable, envir_bin) :
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
    
    def get_numpy_input_envir_set(self, df, envir_variables) :
        # loading channels data in numpy for CNN 
        
        input_envir_set = np.empty_like(np.expand_dims(df[envir_variables[0]].values, axis = 1))
        for env_var in envir_variables :
            input_envir = np.expand_dims(df[env_var].values, axis = 1)
            input_envir_set = np.append(input_envir_set, input_envir, axis = 1)
        input_envir_set = np.delete(input_envir_set, 0, axis=1)
        return input_envir_set

    def get_numpy_input_channel_set(self, df, channels) :
        # loading channels data in numpy for CNN 

        input_channel_set = np.empty_like(np.expand_dims(df[channels[0]].values, axis = 2))
        for channel in channels :
            input_channel = np.expand_dims(df[channel].values, axis = 2)
            input_channel_set = np.append(input_channel_set, input_channel, axis = 2)
        input_channel_set = np.delete(input_channel_set, 0, axis=2)
        return input_channel_set