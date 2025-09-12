from typing import Union
import numpy as np

class UnequalShapesException (Exception):
    def __init__ (self, X: np.ndarray, Y: np.ndarray):
        self.X
        self.Y
        self.use_message = "[-] Error: Unequal shapes of dataset's X and Y -> X: Row {} | Column: {}, Y: Row: {} | Column: {}"

    def __str__ (self):
        return self.use_message.format(self.X.shape[0], self.X.shape[1], self.Y.shape[0], self.Y.shape[1])

class UnequalDatatypesException (Exception):
    def __init__ (self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
        self.ude_message = "[-] Error: Unequal dtypes. X dtype: {} | Y dtype: {}"

    def __str__ (self):
        return self.ude_message.format(self.X.dtype, self.Y.dtype)

class InfinityException (Exception):
    def __init__ (self, X: Union[np.ndarray | list[np.ndarray]]):
        self.X = X
        self.ie_message = "[-] Error: One of the datasets has infinity value"

    def __str__ (self):
        if isinstance(X, list):
            if any(np.isinf(dataset) for dataset in X):
                return self.ie_message 
