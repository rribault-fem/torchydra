
import xarray as xr
import numpy as np
import pickle
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import logging
from config_schema import PipeConfig

from preprocessing.Preprocessor_Zefyros_dataset_ANN import  get_cos_sin_decomposition
from preprocessing.envir_scaler import Envir_scaler
from preprocessing.decomp_y_spectrum import Decomp_y_spectrum
from preprocessing.y_spectrum_scaler import Y_spectrum_scaler
from preprocessing.split_transform import Split_transform
from models.surrogate.surrogate import SurrogateModel

from sklearn.utils import shuffle

log = logging.getLogger('train_surrogate')
cs = ConfigStore.instance()
cs.store(name="config_schema", node=PipeConfig)

@hydra.main(config_path="configs", config_name="config.yaml")

def main(cfg :  DictConfig):
        """
        This function serves as the main entry point for the script. It takes in a configuration object and uses it to train a surrogate model.

        The function first creates an instance of the `PipeConfig` class using the provided configuration. It then pre-processes the data using the `Pre_process_data` function.

        After pre-processing, the pipeline is saved for future use. The specified model type is then imported and trained on the pre-processed data using the `Train_model` function.

        Finally, after training, both the pipeline and trained model are saved for future use.

        Args:
        cfg (DictConfig): The configuration object used to specify training parameters.

        Returns:
        None
        """

        pipe = PipeConfig(**cfg)
        
        # Pre-process data
        x_train, y_train, x_test, y_test = Pre_process_data(pipe)
        
        # save the pipeline for future use and inverse transform
        log.info("Saving pipeline")
        file_path = 'pipeline.pkl'
        with open(file_path, 'wb') as f:
                pickle.dump(pipe, f)

        # import  model type
        import importlib
        surrogate = importlib.import_module("models."+pipe.model.model_type)
        surrogate = getattr(surrogate, pipe.model.model_type)
        # Train model
        log.info(f"Training {pipe.model.model_type} model")
        Train_model(pipe, surrogate, x_train, y_train, x_test, y_test)

        # Save model
        surrogate.save_model(pipe, model)



def Pre_process_data(pipe : PipeConfig):
        """
        This function pre-processes the data before training. It takes in an instance of the `PipeConfig` class and uses it to perform various operations on the data.

        The function first loads the dataset and drops any missing values. It then creates a dictionary to store the units of each variable.

        Next, direction columns are rearranged to fit with neural networks. This is done using the `get_cos_sin_decomposition` function.

        After rearranging direction columns, the data is split into training and testing sets using the specified split transform. The input data is then scaled using the specified scaler.

        Args:
        pipe (PipeConfig): An instance of the `PipeConfig` class containing configuration information.

        Returns:
        tuple: A tuple containing four elements: X_train, X_test, Y_train, Y_test.
        """
        ####
        #Start pipeline
        ####
        df = xr.open_dataset(pipe.paths.dataset)
        df = df.dropna(dim='time', how='any')

        unit_dictionnary = {}
        for var in pipe.inputs_outputs.envir_variables :
                unit_dictionnary[var] = df[var].attrs['unit']
        for var in pipe.inputs_outputs.neuron_variables :
                unit_dictionnary[var] = df[var].attrs['unit']

        ####
        # Re-arrange direction columns because 0/360 discontinuity do not fit with neural networks.
        ####
        for dict_key in [{'mag10':'theta10'}, {'hs':'dp'}] : #, {'cur':'cur_dir'}] :
                df = get_cos_sin_decomposition(dict_key, df)
                for magnitude, angle in dict_key.items() :
                        pipe.inputs_outputs.envir_variables.remove(angle)
                        pipe.inputs_outputs.envir_variables.append(f'{magnitude}_cos')
                        pipe.inputs_outputs.envir_variables.append(f'{magnitude}_sin')
        
        ####
        # Split data into train and test sets. 
        ####
        log.info(f"Instantiating split_transform <{pipe.preprocessing.split_transform._target_}>")
        split_transform: Split_transform = hydra.utils.instantiate(pipe.preprocessing.split_transform)
        X_train, X_test, Y_train, Y_test = split_transform.process_data(df=df, 
                                                                        X_channel_list=pipe.inputs_outputs.envir_variables,
                                                                        Y_channel_list=pipe.inputs_outputs.neuron_variables,
                                                                        df_train_set_envir_filename=pipe.paths.training_env_dataset)
        ####
        # Scale input data with scaler defined in hydra config file
        ####
        log.info(f"Instantiating envir_scaler <{pipe.preprocessing.envir_scaler._target_}>")
        envir_scaler: Envir_scaler = hydra.utils.instantiate(pipe.preprocessing.envir_scaler)
        x_train, x_test  = envir_scaler.scale_data(X_train, X_test)

        ####
        # Decompose y data with decomposition methode defined in hydra config file
        #### 
        if pipe.preprocessing.perform_decomp :
                log.info(f"Instantiating decomp_y_spectrum <{pipe.preprocessing.decomp_y_spectrum._target_}>")
                decomp: Decomp_y_spectrum = hydra.utils.instantiate(pipe.preprocessing.decomp_y_spectrum)
                Y_train, Y_test = decomp.decomp_data(Y_train, Y_test)

        ####
        # Scale Y spectrum data with scaler defined in hydra config file
        ####
        log.info(f"Instantiating y_spectrum_scaler <{pipe.preprocessing.y_spectrum_scaler._target_}>")
        y_spectrum_scaler: Y_spectrum_scaler = hydra.utils.instantiate(pipe.preprocessing.y_spectrum_scaler)
        y_train, y_test = y_spectrum_scaler.scale_data(Y_train, Y_test)

        ####
        # Shuffle training data
        ####
        x_train, y_train = shuffle(x_train, y_train)


        log.info(f'x_train shape: {np.shape(x_train)}')
        log.info(f'y_train shape: {np.shape(y_train)}')

        return x_train, y_train, x_test, y_test



def Train_model(pipe : PipeConfig, x_train : np.ndarray, y_train : np.ndarray, x_test : np.ndarray, y_test : np.ndarray, surrogate : SurrogateModel, x_scaler, y_scalers, unit_dictionnary, y_pcas) -> None:

        # Train model
        surrogate = surrogate(np.shape(y_train)[2], np.shape(x_train)[1])
        surrogate.dataset_path = pipe.paths.training_env_dataset
        surrogate.input_variables = pipe.inputs_outputs.envir_variables
        surrogate.output_variables = pipe.inputs_outputs.neuron_variables
        surrogate.Frequency_psd(pipe.preprocessing.cut_low_frequency)

        surrogate.set_units_dictionnary(unit_dictionnary)
        surrogate.set_modelname(pipe.model.name, pipe.model.version)

        return surrogate

        


if __name__ == "__main__":
    main()