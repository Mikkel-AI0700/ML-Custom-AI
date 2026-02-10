from typing import Union, Any
import numpy as np

class NonExistentDataset (Exception):
    """Raised when a required dataset input is missing.

    Parameters
    ----------
    X : np.ndarray
        Feature/primary dataset.
    Y : np.ndarray or None
        Optional target/secondary dataset.

    Attributes
    ----------
    X : np.ndarray
        Feature/primary dataset.
    Y : np.ndarray or None
        Optional target/secondary dataset.
    """
    def __init__ (self, X: np.ndarray, Y: Union[np.ndarray | None]):
        self.X = X
        self.Y = Y
        self.ned_message = "[-] Error: Either X and/or Y is None -> X: {} | Y: {}"

    def __str__ (self):
        return self.ned_message.format(type(self.X), type(self.Y))

class UnequalShapesException (Exception):
    """Raised when datasets have unequal leading dimensions.

    Parameters
    ----------
    X : np.ndarray
        Feature dataset.
    Y : np.ndarray
        Target/label dataset.

    Attributes
    ----------
    X : np.ndarray
        Feature dataset.
    Y : np.ndarray
        Target/label dataset.
    """
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

class Not1DDataset (Exception):
    """Raised when an input dataset is not 1-dimensional.

    Parameters
    ----------
    X : np.ndarray
        Dataset that was expected to be 1D.

    Attributes
    ----------
    X : np.ndarray
        Dataset that was expected to be 1D.
    """
    def __init__ (self, X: np.ndarray):
        self.X = X
        self.unequal_1d = "[-] Error: Passed dataset is not 1D, convert to 1D. X shape: {}"

    def __str__ (self):
        return self.unequal_1d.format(self.X.shape)

class Not2DDataset (Exception):
    """Raised when an input dataset is not 2-dimensional.

    Parameters
    ----------
    X : np.ndarray
        Dataset that was expected to be 2D.

    Attributes
    ----------
    X : np.ndarray
        Dataset that was expected to be 2D.
    """
    def __init__ (self, X: np.ndarray):
        self.X = X
        self.unequal_2d = "[-] Error: Passed dataset is not 2D, convert to 2D. X shape: {}"

    def __str__ (self):
        return self.unequal_2d.format(self.X.shape)

class InfinityException (Exception):
    """Raised when input data contains positive/negative infinity values."""
    def __init__ (self):
        self.ie_message = "[-] Error: Training matrix X or column vector Y contains positive/negative infinity values."

    def __str__ (self):
        return self.ie_message

class NaNException (Exception):
    """Raised when input data contains NaN values."""
    def __init__ (self):
        self.nan_message = "[-] Error: Training matrix X or column vector Y contains NaN values."

    def __str__ (self):
        return self.nan_message