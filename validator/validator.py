import inspect
from typing import Callable
import numpy as np
import pandas as pd

# Global constant
EXIT_FAILURE = 1
EXIT_SUCCESS = 0

# Dataset type, shape and existence validators
class DatasetValidation:
    def __init__ (self):
        self.NULL_DATASET_ERROR = "Error: One or all of the datasets are null"
        self.UNEQUAL_SHAPES_ERROR = "Error: The shapes of two datasets don't match"
        self.UNEQUAL_DATATYPES_ERROR = "Error: The datatypes of the two datasets isn't equal. Can't perform operations"

    def validate_existence (self, datasets: list[np.ndarray]):
        try:
            if any(len(dset) <= 0 for dset in datasets):
                raise ValueError(self.NULL_DATASET_ERROR)
            else:
                return True
        except TypeError as null_dataset_error:
            print(null_dataset_error)
            exit(EXIT_FAILURE)

    def validate_shapes (self, dataset_x: np.ndarray, dataset_y: np.ndarray):
        try:
            if dataset_x.shape[0] != dataset_y.shape[0]:
                raise ValueError(self.UNEQUAL_SHAPES_ERROR)
            else:
                return True
        except ValueError as unequal_shapes_error:
            print(unequal_shapes_error)
            exit(EXIT_FAILURE)

    def validate_types (self, dataset_x: np.ndarray, dataset_y: np.ndarray):
        try:
            if dataset_x.dtype != dataset_y.dtype:
                raise TypeError(self.UNEQUAL_DATATYPES_ERROR)
            else:
                return True
        except TypeError as unequal_datatypes_error:
            print(unequal_datatypes_error)
            exit(EXIT_FAILURE)

class ClassificationMetricValidation:
    def __init__ (self):
        self.BINARY_LIMIT = 2
        self.ZERO_DIVISION_ERROR = "Error: Your model hasn't predicted anything \"correct\""
        self.MULTI_ON_BINARY_ERROR = "Error: Average parameter is set to binary but classes is multiclass"

    def _return_function_arguments (self, metric_function: Callable):
        return inspect.getfullargspec(metric_function)

    def check_label_count (self, y_true: np.ndarray, classif_metric: Callable):
        try:
            args_tuple = self._return_function_arguments(classif_metric)
            if np.unique(y_true) > self.BINARY_LIMIT:
                pass
        except:
            pass

    def check_parameters (self):
        pass

class RegressionMetricValidation:
    def __init__ (self):
        pass

class ParameterValidation:
    pass
