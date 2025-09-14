from typing import Union
import numpy as np

class UnequalAlignmentException (Exception):
    def __init__ (self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

class UnequalShapesException (Exception):
    def __init__ (self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = X
        self.use_message = (
            "[-] Error: Unequal shapes of both datasets"
            f"X rows: {self.X.shape[0]} | X columns: {self.X.shape[1]}"
            f"Y rows: {self.Y.shape[0]} | Y columns: {self.Y.shape[1]}"
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
