from dataclasses import dataclass
from typing import List, Any, Optional, Dict
from preprocessing.envir_scaler import Envir_scaler
from preprocessing.decomp_y_spectrum import Decomp_y_spectrum
from preprocessing.y_spectrum_scaler import Y_spectrum_scaler	
from preprocessing.split_transform import Split_transform
from preprocessing.feature_eng import FeatureEng
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from typing import Any

@dataclass
class Inputs_ouputs :
    envir_variables : List[str]
    neuron_variables : List[str]

@dataclass
class Preprocessing :    
        paths : Dict
        inputs_outputs : Inputs_ouputs
        feature_eng : FeatureEng
        split_transform : Split_transform
        envir_scaler : Envir_scaler
        perform_decomp : bool
        decomp_y_spectrum: Decomp_y_spectrum
        y_spectrum_scaler : Y_spectrum_scaler
        cut_low_frequency: float


@dataclass
class Model :
    model_type : str
    name : str
    version : str
    dropout_rate : float

@dataclass
class Trainer :
    epochs : int
    batch_size : int
    loss : str
    learning_rate : float
