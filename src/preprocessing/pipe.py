from dataclasses import dataclass
from typing import List

@dataclass
class Inputs_ouputs :
    envir_variables : List[str]
    neuron_variables : List[str]

@dataclass
class Paths :
    dataset: str
    training_env_dataset : str

@dataclass
class Preprocessing :
    envir_direction_dict : dict
    envir_bin : dict
    perform_pca : bool
    pca_n_component: int
    cut_low_frequency: float

@dataclass
class Train_test_split :
    test_nb : int
    cluster: int

@dataclass
class Pipe_Surrogate :
    inputs_outputs : Inputs_ouputs
    paths : Paths
    preprocessing : Preprocessing
    train_test_split : Train_test_split

