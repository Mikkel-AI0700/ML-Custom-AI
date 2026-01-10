from typing import Union, Any
import numpy as np

class NonExistentDataset (Exception):
    def __init__ (self, non_existent_datasets: list[Any]):
        self.non_existent_datasets = non_existent_datasets
        self.ned_message = (
            "[-] Error: A dataset with either a None type or other datatype is present"
            "Datasets with None or other datatypes: {}"
        )

    def __str__ (self):
        return self.ned_message.format(self.non_existent_datasets)

class UnequalAlignmentException (Exception):
    def __init__ (self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
        self.uae_message = (
            "[-] Error: The row length of both datasets X and Y don't match"
            "X row length: {} | Y row length: {}"
        )

    def __str__ (self):
        return self.uae_message.format(self.X.shape[0], self.Y.shape[0])

class UnequalShapesException (Exception):
    def __init__ (self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
        self.use_message = (
            "[-] Error: Unequal shapes of both datasets"
            "X rows: {} | X columns: {}"
            "Y rows: {} | Y columns: {}"
        )

    def __str__ (self):
        return self.use_message.format(
            self.X.shape[0], 
            self.X.shape[1], 
            self.Y.shape[0], 
            self.Y.shape[1]
        )

class UnequalDatatypesException (Exception):
    def __init__ (self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
        self.ude_message = "[-] Error: Unequal dtypes. X dtype: {} | Y dtype: {}"

    def __str__ (self):
        return self.ude_message.format(self.X.dtype, self.Y.dtype)
    
class Not1DDataset (Exception):
    def __init__ (self, X: np.ndarray):
        self.X = X
        self.unequal_1d = "[-] Error: Passed dataset is not 1D, convert to 1D. X shape: {}"

    def __str__ (self):
        return self.unequal_1d.format(self.X.shape)

class Not2DDataset (Exception):
    def __init__ (self, X: np.ndarray):
        self.X = X
        self.unequal_2d = "[-] Error: Passed dataset is not 1D, convert to 2D. X shape: {}"

    def __str__ (self):
        return self.unequal_2d.format(self.X.shape)

class InfinityException (Exception):
    def __init__ (self, infinity_datasets: list):
        self.inf_datasets = infinity_datasets
        self.ie_message = "[-] Error: The following datasets has infinity values: {}"

    def __str__ (self):
        return self.ie_message.format(self.inf_datasets)

class NaNException (Exception):
    def __init__ (self, nan_datasets: list):
        self.nan_datasets = nan_datasets
        self.nan_message = "[-] Error: The following datasets has NaN values: {}"

    def __str__ (self):
        return self.ie_message.format(self.nan_datasets)
