#pass Pytho SL
from typing import Union

# Python third-party imports
import numpy as np
import pandas as pd
import scipy as scp

# Python local imports
from validator.validator import Validate
from metrics.classification.classif_metrics import Accuracy

class LogisticRegression:
    def __init__ (self, epochs: int = 1000, fit_intercept: bool = True):
        self.partial_derivative_m = None
        self.partial_derivative_b = None
        self.fit_intercept = fit_intercept
        self.epochs = epochs

    def _initialize_weights (self, train_x: np.ndarray):
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            return np.hstack([np.ones((train_x.shape[0], 1)), train_x])
        else:
            self.partial_derivative_b = 0.0

    def _compute_weights_derivative (
        self,
        train_x: np.ndarray,
        train_y: np.ndarray
    ):
        pass

    def _compute_bias_derivative (self):
        pass

    def _update_weights_derivatives (self):
        pass

    def _update_bias_derivatives (self):
        pass

    def _sigmoid_function (self, pred_y: np.ndarray):
        pass

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame]
    ):
        for epoch in range(self.epochs):
            pass

    def predict (
        self,
        test_x: Union[np.ndarray | pd.DataFrame]
    ):
        pass

