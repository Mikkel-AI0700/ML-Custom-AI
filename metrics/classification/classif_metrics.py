from typing import Union
import numpy as np
import pandas as pd
from validator.validator import DatasetValidation, ClassificationMetricValidation

dset_val = DatasetValidation()
classif_val = ClassificationMetricValidation()

def accuracy (true_y: Union[np.ndarray | pd.DataFrame], pred_y: Union[np.ndarray | pd.DataFrame]):
    pass

def precision (true_y: Union[np.ndarray | pd.DataFrame], pred_y: Union[np.ndarray | pd.DataFrame]):
    pass

def recall (true_y: Union[np.ndarray | pd.DataFrame], pred_y: Union[np.ndarray | pd.DataFrame]):
    pass

def f1 (true_y: Union[np.ndarray | pd.DataFrame], pred_y: Union[np.ndarray | pd.DataFrame]):
    pass

