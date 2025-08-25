from typing import Union
import numpy as np
import pandas as pd
import scipy as scp
from validator.validator import DatasetValidation
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
        self.validator = DatasetValidation()

    def _initialize_weights (self, train_x: np.ndarray):
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            return np.hstack([np.ones((train_x.shape[0], 1)), train_x])
        else:
            self.partial_derivative_b = 0.0

    def _compute_weights_derivative (self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
        return 1 / len(train_x) * np.sum((pred_y - train_y)[:, np.newaxis] * train_x, axis=0)

    def _compute_bias_derivative (self, train_y: np.ndarray, pred_y: np.ndarray):
        return 1 / len(train_x) * np.sum(pred_y - train_y)

    def _update_weights_derivatives (self, computed_weights_gradients: np.ndarray):
        self.partial_derivative_m = self.partial_derivative_m - self.learning_rate * computed_weights_gradients

    def _update_bias_derivatives (self, computed_bias_gradients: np.ndarray):
        self.partial_derivative_b = self.partial_derivative_m - self.learning_rate * computed_bias_gradients

    def _sigmoid_function (self, pred_y: np.ndarray):
        return 1 / 1 + np.e ** -(pred_y)

    def fit (self, train_x: Union[np.ndarray | pd.DataFrame], train_y: Union[np.ndarray | pd.DataFrame]):
        if (self.validator.validate_existence([train_x, train_y]) and
            self.validator.validate_shapes(train_x, train_y)
        ):
            pass

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch} | M: {self.partial_derivative_m} | B: {self.partial_derivative_b}")

            # Training inference and loss calculating
            predictions = np.dot(train_x, self.partial_derivative_m)
            sigmoid_probabilities = self._sigmoid_function(predictions)
            computed_cost = log_loss(train_y, sigmoid_probabilities)

            # Gradient computation and updating
            computed_weights = self._compute_weights_derivative(train_x, train_y, predictions)
            computed_bias = self._compute_bias_derivative(train_x, train_y, predictions)
            self._update_weights_derivatives(computed_weights)
            self._update_bias_derivatives(computed_bias)

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        predictions = np.dot(self.partial_derivative_m, test_x)

