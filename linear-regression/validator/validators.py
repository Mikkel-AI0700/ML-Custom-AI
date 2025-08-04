from typing import Union
import numpy as np
import pandas as pd

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

class Validate:
    def __init__ (self):
        self.NULL_DATASET_ERROR = "[-] Error: One of the datasets is null"
        self.UNEQUAL_DATASET_SHAPES = "[-] Error: Dataset shapes are unequal"

    def validate_existence (self, datasets: list[Union[np.ndarray | pd.DataFrame]]):
        try:
            if not any(isinstance(dset, (np.ndarray, pd.DataFrame)) for dset in datasets):
                raise TypeError(self.NULL_DATASET_ERROR)
            else:
                return True
        except TypeError as null_dataset_error:
            print(null_dataset_error)
            exit(EXIT_FAILURE)

    def validate_shapes (
        self,
        dataset_x: Union[np.ndarray | pd.DataFrame],
        dataset_y: Union[np.ndarray | pd.DataFrame]
    ):
        try:
            if dataset_x.shape[0] != dataset_y.shape[0]:
                raise ValueError(self.UNEQUAL_DATASET_SHAPES)
            else:
                return True
        except ValueError as unequal_dataset_shapes:
            print(unequal_dataset_shapes)
            exit(EXIT_FAILURE)

