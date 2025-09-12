from typing import Union, Callable
from inspect import signature
import numpy as np
import pandas as pd
from errors.DatasetErrors import (
    UnequalShapesException,
    UnequalDatatypesException
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

class DatasetValidation:
    def __init__ (self):
        self.UNEQUAL_SHAPES_ERROR = "{} -> X: Row: {} | Column: {}, Y: Row: {} | Column: {}"
        self.UNEQUAL_DATATYPES_ERROR = "{} -> X dtype: {} | Y dtype: {}"

    def check_shapes (self, X: np.ndarray, Y: np.ndarray):
        try:
            if X.shape[0] != Y.shape[0]:
                raise UnequalShapesException(self.UNEQUAL_SHAPES_ERROR)
            else:
                return True
        except UnequalShapesException as use_message:
            print(use_message.format(use_message, X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]))
            exit(EXIT_FAILURE)

    def check_datatypes (self, X: np.ndarray, Y: np.ndarray):
        try:
            if X.dtype != Y.dtype:
                raise UnequalDatatypesException(self.UNEQUAL_DATATYPES_ERROR)
            else:
                return True
        except UnequalDatatypesException as ude_message:
            print(ude_message.format(ude_message, X.dtypes, Y.dtypes))
            exit(EXIT_FAILURE)

