# Deep_learning_model_training
# Surrogate Model Training
This script trains a surrogate model using the specified configuration.

## Configuration
The configuration is stored in a YAML file and can be modified to change the behavior of the training process. 
The management of YAML file is based on the hydra package : https://hydra.cc/docs/intro/
Please perform the structured config hydra tutorial before using this package.
https://hydra.cc/docs/1.0/tutorials/structured_config/intro/

The `PipeConfig` class is used to store the configuration. 
This class is defined in config_schema.py and is used to validate the configuration YAML file.

## Outputs
The trained model, logs, experiment configuration and environments used during training are saved in the `outputs` directory.
One subfolder is created for each taskname define in the config.yaml file.
For each experiment, a subfolder is created with the current date and time.
This behavior is defined in the configs/hydra/default.yaml file.

## Usage
To run this script, use the following command: `python train_surrogate.py`
This will train a surrogate model using the specified configuration and save it for future use.

## Pre-processing

The `Pre_process_data` function is used to pre-process the data before training. It loads the dataset and performs various operations such as dropping missing values and rearranging direction columns.

## Training

The `Train_model` function is used to train the surrogate model. It imports the specified model type and trains it on the pre-processed data.

## Saving

After training, both the pipeline and trained model are saved for future use.


## Configuration example
Several scaler and decomposition methods are available : all scalers from scikit-learn and the decomposition methods from scikit-learn. 
for example you can change the environmental scaler type, edit the `scaler_options` of the envir_scaler object to any of the following:
- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- ...

If required you can also implement you own scaler method in envir_scaler.py.

To split a dataset into a test and training data set you can either used the proposed `find_test_set_in_model_validity_domain` or write another method in split_transform.py and set the method name in `split_method` parameter of `config.yaml` file.

To change the configuration, edit the `config.yaml` file.
--> To change a model type, edit the `model_type` parameter.
