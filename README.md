# Introduction

✅ This package is used to experiment several deep learning models, save on boilerplate code.
Easily add new models, datasets, experiments, preprocessing steps.
By using this template, you can focus on deep learning architecture, hyperparameters tuning and share models and experiments with collegues.

It is based on an opensource project template https://github.com/ashleve/lightning-hydra-template and uses the following packages:

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

--> be familiar with these packages before using this project.


## Installation

#### Pip

```bash
# clone project
git clone https://github.com/rribault-fem/torchydra
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```
#### Conda

```bash
# clone project
git clone https://github.com/rribault-fem/torchydra
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

#### DVC

```bash
# from project root directory, recover the shared models and datasets
DVC pull
```

#### env.yaml file

select the path dedicated to your environment in the env.yaml file.

```bash	
# example of file for storing private and user specific environment variables, like keys or system paths
# rename it to ".env" (excluded from version control by default)
# .env is loaded by train.py automatically
# hydra allows you to reference variables in .yaml configs with special syntax: ${oc.env:MY_VAR}

# refer the dataset to use for training. Note that datasets are version controlled by DVC in the data/netcdf_databases folder
DATASET_PATH: C:\Users\romain.ribault\Documents\GitHub\torchydra\data\netcdf_databases\reference_dataset_Zefyros_windeurope.nc

# for inference with subsee4D path structure (FEM specific)
INFER_SAVE_PATH: \\10.12.89.104\zefyros_calc\PreProd\storage

# Refer the model and preprocessing parameters to use for inference:
INFER_EXPERIMENT_PATH: C:\Users\romain.ribault\Documents\GitHub\Deep_learning_model_training\outputs\train_surrogate\runs\2023-04-18_09-19-57
```

## How to run

Select the output sub-directory by overriding the taks_name in train.yaml file.
Describe your experiment with a tag in the train.yaml file.

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)


```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

Search for best hyperparameters with [Optuna](https://optuna.org/)

```bash
python src/train.py -m hparams_search=optuna
```
python src/train.py -m hparams_search=optuna

## Outputs

The trained model, logs, experiment configuration and environments used during training are saved in the `outputs` directory.
The output sub-directory can be changed by overriding the taks_name in train.yaml file.
For each run, a subfolder is created with the current date and time.
This behavior is defined in the configs/hydra/default.yaml file.

Please define tags in the train.yaml file to easily identify your experiments.

<br>

## Commit results

Once you have trained your model, and get usefull results you can commit the models and dataprocessing objects to the DVC repository.

```bash
# add the results to the DVC repository
dvc push
```

## Project Structure

The directory structure of new project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Ligthning DataModule configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs (Tensorboad...)
│   ├── model                    <- Model configs
│   ├── preprocessing            <- Project preprocessing configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── outputs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   ├── preprocessing            <- Preprocessing scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```
<br>

## Pre-processing

The `Pre_process_data` function is used to pre-process the data before training.
It loads the dataset and performs various operations such as dropping missing values and rearranging direction columns.

You can modify the pre-processing steps in the `configs\preprocessing\default.yaml` file.

For example :
Several scaler are available : all scalers from scikit-learn from scikit-learn. 
for example you can change the environmental scaler type, edit the `envir_scaler.scaler._target_` of the envir_scaler object to any of the following:
- sklearn.preprocessing.StandardScaler   
- sklearn.preprocessing.MinMaxScaler
- sklearn.preprocessing.MaxAbsScaler
- ...

Same applmies with the `y_spectrum_scaler` object.
The scaler for measurement channels (y) changed by modifying the y_spectrum_scaler.scaler_options name.


You can change the feature engineering method to a new function by changing the name of the 'sin_cos_method'.

The split_method can be changed by defining a new function in the `src\preprocessing\split_transform.py` file and changing the name of the 'split_method' in the `configs\preprocessing\default.yaml` file.

You can choose to perform or not perform a decomposition on the measurement channels (y) by changing the `perform_decomp` to True or False.
The type of decomposition shall be defined in decomp_y_spectrum.decomp_option. It shall be sklearn.decomposition object.


## Training

The `Train_model` function is used to train the surrogate model. It imports the specified model type and trains it on the pre-processed data.

## Saving

After training, both the pipeline and trained model are saved for future use.

## Visualise training results

To visualise the training results, you can use the [tensorboard](https://www.tensorflow.org/tensorboard) tool.
Tensorboard is the default tool but others are available (cf lightning / hydra ashleve documentation)

```bash
# from project root directory
tensorboard --logdir=outputs/train

```

## Configuration example

If required you can also implement you own scaler method in envir_scaler.py.

To split a dataset into a test and training data set you can either used the proposed `find_test_set_in_model_validity_domain` or write another method in split_transform.py and set the method name in `split_method` parameter of `config.yaml` file.

To change the configuration, edit the `config.yaml` file.
--> To change a model type, edit the `model_type` parameter.

## Add a new model type for the surrogate model
To add a new model type for the surrogate model, you need to create a new class in the `src/models/components` folder.
It shall be a subclass of `nn.Module` and implement the `forward` method.
refer to the available examples.

Once the class created you need to change the `_target_` parameter in the `model_net` section of the `train.yaml` file.

If needed you can create a new config file in `configs/model_net` to define new parameters for your model.


## License

Lightning-Hydra-Template is licensed under the MIT License.

```
MIT License

Copyright (c) 2021 ashleve
Copyright (c) 2023 rribault

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```