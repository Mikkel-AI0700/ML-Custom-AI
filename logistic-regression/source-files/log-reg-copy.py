from typing import Union
import numpy as np
import pandas as pd
import scipy as scp
from validator.validator import Validate
from metrics.classification.classif_metrics import log_loss

class LogisticRegression:
    def __init__ (
        self, 
        learning_rate: Union[int | float] = 0.0001, 
        epochs: int = 1000, 
        fit_intercept: bool = True
    ):
        self.partial_derivative_m = None
        self.partial_derivative_b = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.validator = Validate()

    def _initialize_weights (self, train_x: np.ndarray):
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            return np.hstack([np.ones((train_x.shape[0], 1)), train_x])
        else:
            self.partial_derivative_b = 0.0

    def _compute_weights_derivative (self, train_x: np.ndarray, train_y: np.ndarray,pred_y: np.ndarray):
        return -2 / len(train_x) * np.sum(train_x * (train_y - pred_y))

    def _compute_bias_derivative (self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
        return -2 / len(train_x) * np.sum(train_y - pred_y)

    def _update_weights_derivatives (self, computed_weights_gradients: np.ndarray):
        self.partial_derivative_m = self.learning_rate - computed_weights_gradients

    def _update_bias_derivatives (self, computed_bias_gradients: np.ndarray):
        self.partial_derivative_b = self.learning_rate - computed_bias_gradients

    def _sigmoid_function (self, pred_y: np.ndarray):
        return 1 / 1 + np.e ** -pred_y

    def fit (self, train_x: Union[np.ndarray | pd.DataFrame], train_y: Union[np.ndarray | pd.DataFrame]):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch} | M: {self.partial_derivative_m} | B: {self.partial_derivative_b}")

            # Training inference and loss calculating
            predictions = np.dot(train_x, self.partial_derivative_m)


            # Gradient computation and updating
            computed_weights = self._compute_weights_derivative(train_x, train_y, predictions)
            computed_bias = self._compute_bias_derivative(train_x, train_y, predictions)
            self._update_weights_derivatives(computed_weights)
            self._update_bias_derivatives(computed_bias)

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        pass

