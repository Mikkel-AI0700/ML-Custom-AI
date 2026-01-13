from typing import Union, Callable
from inspect import signature
import numpy as np
import pandas as pd
from errors.DatasetErrors import (
    NonExistentDataset,
    Not1DDataset,
    Not2DDataset,
    UnequalShapesException,
    UnequalDatatypesException,
    InfinityException,
    NaNException
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

class DatasetValidation:
    """Validate dataset existence, shapes, dtypes, and numeric sanity.

    The validator collects references to datasets that fail specific checks
    (non-existent / non-NumPy, containing infinities, containing NaNs). Public
    validation methods print errors and terminate the process on failure.

    Attributes:
        datasets_with_infinity (list[np.ndarray]): Datasets found (or reported) to contain `inf`.
        datasets_with_nan (list[np.ndarray]): Datasets found (or reported) to contain `NaN`.
        datasets_with_non (list[np.ndarray]): Inputs found (or reported) to be non-existent/non-NumPy.
    """

    def __init__ (self):
        self.datasets_with_infinity = []
        self.datasets_with_nan = []
        self.datasets_with_non = []

    def __check_non (self, X: list[np.ndarray]):
        """Record datasets that are not NumPy arrays.

        Args:
            X (list[np.ndarray]): A list of dataset-like objects to inspect.

        Notes:
            This helper appends any non-`np.ndarray` items to `datasets_with_non`.
        """
        for dataset in X:
            if not any(isinstance(dataset, np.ndarray)):
                self.datasets_with_non.append(dataset)

    def __check_inf (self, X: list[np.ndarray]):
        """Record datasets that contain infinity values.

        Args:
            X (list[np.ndarray]): A list of NumPy arrays to inspect.

        Notes:
            This helper appends any datasets that are detected to contain `inf`
            to `datasets_with_infinity`.
        """
        for dataset in X:
            if np.isinf(dataset):
                self.datasets_with_infinity.append(dataset)

    def __check_nan (self, X: list[np.ndarray]):
        """Record datasets that contain NaN values.

        Args:
            X (list[np.ndarray]): A list of NumPy arrays to inspect.

        Notes:
            This helper appends any datasets that are detected to contain NaNs
            to `datasets_with_nan`.
        """
        for dataset in X:
            if np.isnan(dataset):
                self.datasets_with_nan.append(dataset)

    def check_existence (self, X: list[np.ndarray], Y: None):
        """Check that provided datasets exist and are NumPy arrays.

        Args:
            X (list[np.ndarray]): One or more datasets to validate.
            Y (None): Unused parameter (kept for a uniform validation-call signature).

        Returns:
            None: This method does not return a value.

        Raises:
            SystemExit: Exits with `EXIT_FAILURE` after printing a `NonExistentDataset`
                message if any dataset is considered non-existent/non-NumPy.
        """
        try:
            map(self.__check_non, X)
            if len(self.datasets_with_non) > 0:
                raise NonExistentDataset(self.datasets_with_non)
        except NonExistentDataset as ned_message:
            print(ned_message)
            exit(EXIT_FAILURE)

    def check_shapes (self, X: np.ndarray, Y: np.ndarray):
        """Check that `X` and `Y` have the same first dimension length.

        Args:
            X (np.ndarray): Feature dataset.
            Y (np.ndarray): Label/target dataset.

        Returns:
            bool: `True` if `X.shape[0] == Y.shape[0]`.

        Raises:
            SystemExit: Exits with `EXIT_FAILURE` after printing an `UnequalShapesException`
                message if the leading dimensions differ.
        """
        try:
            if X.shape[0] != Y.shape[0]:
                raise UnequalShapesException(X, Y)
            else:
                return True
        except UnequalShapesException as use_message:
            print(use_message)
            exit(EXIT_FAILURE)

    def check_1D (self, X: np.ndarray):
        """Check that `X` is a 1-dimensional NumPy array.

        Args:
            X (np.ndarray): Dataset to validate.

        Returns:
            None: This method does not return a value.

        Raises:
            SystemExit: Exits with `EXIT_FAILURE` after printing a `Not1DDataset`
                message if `X.ndim != 1`.
        """
        try:
            if X.ndim != 1:
                raise Not1DDataset(X)
        except Not1DDataset as dataset_1d_error:
            print(dataset_1d_error)
            exit(EXIT_FAILURE)

    def check_2D (self, X: np.ndarray):
        """Check that `X` is a 2-dimensional NumPy array.

        Args:
            X (np.ndarray): Dataset to validate.

        Returns:
            None: This method does not return a value.

        Raises:
            SystemExit: Exits with `EXIT_FAILURE` after printing a `Not2DDataset`
                message if `X.ndim != 2`.
        """
        try:
            if X.ndim != 2:
                raise Not2DDataset(X)
        except Not2DDataset as dataset_2d_error:
            print(dataset_2d_error)
            exit(EXIT_FAILURE)

    def check_datatypes (self, X: np.ndarray, Y: np.ndarray):
        """Check that `X` and `Y` have the same NumPy dtype.

        Args:
            X (np.ndarray): Feature dataset.
            Y (np.ndarray): Label/target dataset.

        Returns:
            bool: `True` if `X.dtype == Y.dtype`.

        Raises:
            SystemExit: Exits with `EXIT_FAILURE` after printing an `UnequalDatatypesException`
                message if the dtypes differ.
        """
        try:
            if X.dtype != Y.dtype:
                raise UnequalDatatypesException(X, Y)
            else:
                return True
        except UnequalDatatypesException as ude_message:
            print(ude_message.format(ude_message))
            exit(EXIT_FAILURE)

    def infinity_checks (self, X: list[np.ndarray], Y: None):
        """Check that datasets do not contain infinity values.

        Args:
            X (list[np.ndarray]): One or more datasets to validate.
            Y (None): Unused parameter (kept for a uniform validation-call signature).

        Returns:
            bool: `True` if no infinities are detected.

        Raises:
            SystemExit: Exits with `EXIT_FAILURE` after printing an `InfinityException`
                message if any dataset is detected to contain infinity values.
        """
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
        """Check that datasets do not contain NaN values.

        Args:
            X (list[np.ndarray]): One or more datasets to validate.

        Returns:
            bool: `True` if no NaNs are detected.

        Raises:
            SystemExit: Exits with `EXIT_FAILURE` after printing a `NaNException`
                message if any dataset is detected to contain NaN values.
        """
        try:
            map(self.__check_nan, X)
            if len(self.datasets_with_nan) > 0:
                raise NaNException(self.datasets_with_nan)
            else:
                return True
        except NaNException as nan_exception:
            print(nan_exception)
            exit(EXIT_FAILURE)

    def perform_dataset_validation (self, X: np.ndarray, Y: np.ndarray):
        """Run a standard sequence of validation checks for `X` and `Y`.

        The following checks are performed in order:
        existence, shape compatibility, dtype compatibility, infinity checks, NaN checks.

        Args:
            X (np.ndarray): Feature dataset (or datasets, depending on usage in checks).
            Y (np.ndarray): Label/target dataset.

        Returns:
            None: This method does not return a value.

        Raises:
            SystemExit: Exits with `EXIT_FAILURE` if any delegated validation check fails.
        """
        validation_checks_list = (
            self.check_existence,
            self.check_shapes,
            self.check_datatypes,
            self.infinity_checks,
            self.nan_checks
        )

        for validation_check in validation_checks_list:
            validation_check(X, Y)

