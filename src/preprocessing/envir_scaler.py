import logging
import sklearn.preprocessing as sklpre
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Envir_scaler :
    
    scaler_option : str # the name of a method defined below. This name is defined in Hydra config file
    scaler : Optional[TransformerMixin] = None # can be any sklearn transformer or a custom transformer

    def scale_data(self, X_numpy_channels_training_set : np.array, X_numpy_channels_test_set: np.array) -> Tuple[np.array, np.array]:
        
        log = logging.getLogger('train_surrogate')
        # Select user defined Scaler
        log.info('###')
        log.info("select environmental values scaler")
        if hasattr(sklpre, self.scaler_option):
            self.scaler = getattr(sklpre, self.scaler_option)() 
            log.info(f'Using sklearn scaler: {self.scaler_option}')
        #  elif statement for custom scalers with transformer API here
        else :
            raise ValueError(f"No such method: {self.scaler_option}")
        
        # Fit and transform the data
        x_numpy_channels_training_set = self.scaler.fit_transform(X_numpy_channels_training_set)
        x_numpy_channels_val_set = self.scaler.transform(X_numpy_channels_test_set)
        
        log.info('X  data scaled')
        log.info('###')

        return x_numpy_channels_training_set, x_numpy_channels_val_set
    
