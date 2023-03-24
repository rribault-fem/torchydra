# Deep_learning_model_training
# Surrogate Model Training
This script trains a surrogate model using the specified configuration.

## Configuration

The configuration is stored in a YAML file and can be modified to change the behavior of the training process. The `PipeConfig` class is used to store the configuration.

## Outputs
The trained model, logs, experiment configuration and environments used during training are saved in the `outputs` directory.

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
To change the configuration, edit the `config.yaml` file.
--> To change a model type, edit the `model_type` parameter.
--> : To change the environmental scaler type, edit the `scaler_options` of the envir_scaler object.