from typing import Union
import numpy as np
import pandas as pd
from errors.DatasetErrors import (
    NonExistentDataset,
    Not1DDataset,
    Not2DDataset,
    UnequalShapesException,
    InfinityException,
    NaNException
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

class DatasetValidation:
    """Dataset validation utilities.

    Provides a small set of validation helpers for dataset-like inputs used
    throughout the project.

    The methods follow a *fail-fast* approach: on failure they print a
    domain-specific exception message and terminate the process with
    ``EXIT_FAILURE``.

    Notes
    -----
    Context-aware validation
        Not every stage in the project operates on both features and targets.
        For example, inference and transformer pipelines may validate only
        ``X``.

        Several methods therefore accept ``Y`` as ``None`` and interpret that
        as "validate only X". This makes it possible to keep a uniform
        ``(X, Y)`` call signature (consistent architecture/pipeline assembly)
        while still supporting single-input validation.
    """

    def check_existence (self, X: np.ndarray, Y: Union[np.ndarray | None]):
        """Validate that required dataset inputs are present.

        This check is *context-aware*:

        - If ``Y`` is provided (not ``None``), both ``X`` and ``Y`` must be
          present (not ``None``).
        - If ``Y`` is ``None``, only ``X`` is required.

        Parameters
        ----------
        X : numpy.ndarray
            Feature/primary input dataset.
        Y : numpy.ndarray or None
            Optional target/secondary dataset.

            Passing ``None`` indicates that this validation step is being run
            in a context where only ``X`` exists/should be validated (e.g.
            inference or transformer-only pipelines).

        Returns
        -------
        bool
            ``True`` if the required inputs for the given context are present.

        Raises
        ------
        SystemExit
            Exits with ``EXIT_FAILURE`` after printing ``NonExistentDataset`` if
            a required input is missing.

        Notes
        -----
        Despite the name, this method currently checks presence (``is not
        None``) rather than enforcing a strict ``numpy.ndarray`` type.
        """
        try:
            if X is not None and Y is not None:
                return True
            elif X is not None:
                return True
            else:
                raise NonExistentDataset(X, Y)
        except NonExistentDataset as ned_message:
            print(ned_message)
            exit(EXIT_FAILURE)

    def check_shapes (self, X: np.ndarray, Y: np.ndarray):
        """Validate that ``X`` and ``Y`` share the same sample count.

        Parameters
        ----------
        X : numpy.ndarray
            Feature dataset.
        Y : numpy.ndarray
            Target/label dataset.

        Returns
        -------
        bool
            ``True`` if ``X.shape[0] == Y.shape[0]``.

        Raises
        ------
        SystemExit
            Exits with ``EXIT_FAILURE`` after printing ``UnequalShapesException``
            if the leading dimensions differ.
        """
        try:
            if Y is None:
                return True

            if X.shape[0] != Y.shape[0]:
                raise UnequalShapesException(X, Y)
            else:
                return True
        except UnequalShapesException as use_message:
            print(use_message)
            exit(EXIT_FAILURE)

    def check_1D (self, X: np.ndarray, Y: None):
        """Validate that ``X`` is 1-dimensional.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset to validate.
        Y : None
            Placeholder parameter to keep a consistent ``(X, Y)`` validator
            signature. Not used by this check.

        Raises
        ------
        SystemExit
            Exits with ``EXIT_FAILURE`` after printing ``Not1DDataset`` if
            ``X.ndim != 1``.
        """
        try:
            if Y is None:
                return True

            if X.ndim != 1:
                raise Not1DDataset(X)
        except Not1DDataset as dataset_1d_error:
            print(dataset_1d_error)
            exit(EXIT_FAILURE)

    def check_2D (self, X: np.ndarray, Y: None):
        """Validate that ``X`` is 2-dimensional.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset to validate.
        Y : None
            Placeholder parameter to keep a consistent ``(X, Y)`` validator
            signature. Not used by this check.

        Raises
        ------
        SystemExit
            Exits with ``EXIT_FAILURE`` after printing ``Not2DDataset`` if
            ``X.ndim != 2``.
        """
        try:
            if Y is None:
                return True

            if X.ndim != 2:
                raise Not2DDataset(X)
        except Not2DDataset as dataset_2d_error:
            print(dataset_2d_error)
            exit(EXIT_FAILURE)

    def infinity_checks (self, X: np.ndarray, Y: Union[np.ndarray | None]):
        """Validate that inputs do not contain infinity values.

        This check is *context-aware*:

        - If ``Y`` is provided (not ``None``), both ``X`` and ``Y`` are checked.
        - If ``Y`` is ``None``, only ``X`` is checked.

        Parameters
        ----------
        X : numpy.ndarray
            Feature/primary dataset to validate.
        Y : numpy.ndarray or None
            Optional target/secondary dataset.

        Returns
        -------
        bool
            ``True`` if the required inputs for the given context contain no
            infinities.

        Raises
        ------
        SystemExit
            Exits with ``EXIT_FAILURE`` after printing ``InfinityException`` if
            the check detects infinity values in a required input.
        """
        try:
            if (Y is not None and
                not np.any(np.isinf(X)) and
                not np.any(np.isinf(Y))
            ):
                return True
            elif not np.any(np.isinf(X)):
                return True
            else:
                raise InfinityException()
        except InfinityException as ie_exception:
            print(ie_exception)
            exit(EXIT_FAILURE)

    def nan_checks (self, X: np.ndarray, Y: Union[np.ndarray | None]):
        """Validate that inputs do not contain NaN values.

        This check is *context-aware*:

        - If ``Y`` is provided (not ``None``), both ``X`` and ``Y`` are checked.
        - If ``Y`` is ``None``, only ``X`` is checked.

        Parameters
        ----------
        X : numpy.ndarray
            Feature/primary dataset to validate.
        Y : numpy.ndarray or None
            Optional target/secondary dataset.

        Returns
        -------
        bool
            ``True`` if the required inputs for the given context contain no
            NaNs.

        Raises
        ------
        SystemExit
            Exits with ``EXIT_FAILURE`` after printing ``NaNException`` if the
            check detects NaN values in a required input.
        """
        try:
            if (Y is not None and
                not np.any(np.isnan(X)) and
                not np.any(np.isnan(Y))
            ):
                return True
            elif not np.any(np.isnan(X)):
                return True
            else:
                raise NaNException()
        except NaNException as nan_exception:
            print(nan_exception)
            exit(EXIT_FAILURE)

    def perform_dataset_validation (self, X: np.ndarray, Y: Union[np.ndarray | None]):
        """Run the standard dataset validation pipeline.

        The following checks are performed, in order:

        1. Existence / NumPy type checks
        2. Shape compatibility
        3. Dtype compatibility
        4. Infinity checks
        5. NaN checks

        Parameters
        ----------
        X : numpy.ndarray
            Feature dataset.
        Y : numpy.ndarray
            Target/label dataset.

        Notes
        -----
        This method is intended for supervised training where both ``X`` and
        ``Y`` are present. For single-input contexts (e.g. inference), call the
        individual context-aware checks directly (or supply ``Y=None`` where
        supported).

        Raises
        ------
        SystemExit
            Exits with ``EXIT_FAILURE`` if any delegated validation check fails.
        """
        validation_checks_list = (
            self.check_existence,
            self.check_shapes,
            self.infinity_checks,
            self.nan_checks
        )

        for validation_check in validation_checks_list:
            validation_check(X, Y)

