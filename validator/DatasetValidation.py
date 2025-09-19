from typing import Union, Callable
from inspect import signature
import numpy as np
import pandas as pd
from errors.DatasetErrors import (
    NonExistentDatasets
    UnequalShapesException,
    UnequalDatatypesException,
    InfinityException,
    NaNException
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

class DatasetValidation:
    def __init__ (self):
        self.datasets_with_infinity = []
        self.datasets_with_nan = []
        self.datasets_with_non = []

    def __check_non (self, X: list[np.ndarray]):
        for dataset in X:
            if not any(isinstance(dataset, np.ndarray)):
                self.datasets_with_non.append(dataset)

    def __check_inf (self, X: list[np.ndarray]):
        for dataset in X:
            if np.isinf(dataset):
                self.datasets_with_infinity.append(dataset)

    def __check_nan (self, X: list[np.ndarray]):
        for dataset in X:
            if np.isnan(dataset):
                self.datasets_with_nan.append(dataset)

    def check_existence (self, X: list[np.ndarray]):
        try:
            map(self.__check_non, X)
            if len(self.datasets_with_non) > 0:
                raise NonExistentDatasets(self.datasets_with_non)
        except NonExistentDatasets as ned_message:
            print(ned_message)
            exit(EXIT_FAILURE)

    def check_shapes (self, X: np.ndarray, Y: np.ndarray):
        try:
            if X.shape[0] != Y.shape[0]:
                raise UnequalShapesException(X, Y)
            else:
                return True
        except UnequalShapesException as use_message:
            print(use_message)
            exit(EXIT_FAILURE)

    def check_datatypes (self, X: np.ndarray, Y: np.ndarray):
        try:
            if X.dtype != Y.dtype:
                raise UnequalDatatypesException(X, Y)
            else:
                return True
        except UnequalDatatypesException as ude_message:
            print(ude_message.format(ude_message)
            exit(EXIT_FAILURE)

    def infinity_checks (self, X: list[np.ndarray]):
        try:
            map(self.__check_inf, X)
            if len(self.datasets_with_infinity) > 0:
                raise InfinityException(self.datasets_with_infinity)
            else:
                return True
        except InfinityException as ie_exception:
            print(ie_exception)
            exit(EXIT_FAILURE)

    def nan_checks (self, X: list[np.ndarray]):
        try:
            map(self.__check_nan, X)
            if len(self.datasets_with_nan) > 0:
                raise NaNException(self.datasets_with_nan)
            else:
                return True
        except NaNException as nan_exception:
            print(nan_exception)
            exit(EXIT_FAILURE)
