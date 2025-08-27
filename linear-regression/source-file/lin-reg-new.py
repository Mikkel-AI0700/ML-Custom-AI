from typing import Union
import numpy as np
import pandas as pd
from validator.validators import DatasetValidation

class LinearRegression:
    def __init__ (self, epoch: int, learning_rate: float = 1e-4, fit_intercept: bool = True):
        self.partial_derivative_m = None
        self.partial_derivative_b = None
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.fit_intercept = True
        self.validator = DatasetValidator()

    def _initialize_weights_bias (self):
        pass

    def _compute_weights_gradient (self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
        return -(2 / len(train_x)) * np.sum(train_x * (train_y * pred_y))

    def _compute_bias_gradients (self, train_y: np.ndarray, pred_y: np.ndarray):
        return -(2 / len(train_y)) * np.sum(train_y - pred_y)

    def _update_weights_gradient (self, computed_weights_gradients: np.ndarray):
        self.partial_derivative_m -= self.learning_rate * computed_weights_gradients

    def _update_bias_gradient (self, computed_bias_gradients: np.ndarray):
        self.partial_derivative_b -= self.learning_rate * computed_bias_gradients

    def fit (
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame],
    ):
        if (self.validator.validate_existence([train_x, train_y]) and
            self.validator.validate_shapes(train_x, train_y)
        ):
            pass

        if isinstance(train_x, pd.DataFrame):
            train_x = train_x.to_numpy()
        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.to_numpy()

        for epoch in range(self.epochs):
            print(f"Batch: {epoch + 1} | M: {self.partial_derivative_m} | B: {self.partial_derivative_b}")
            
            # Main training loop
            predictions = np.dot(train_x, self.partial_derivative_m)
            computed_weights = self._compute_weights_gradient(train_x, train_y, predictions)
            computed_bias = self._compute_bias_gradients(train_y, predictions)
            self._update_weights_gradient(computed_weights)
            self._update_bias_gradient(computed_bias) 

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        if self.validator.validate_existence([test_x]):
            pass
        if isinstance(test_x, pd.DataFrame):
            test_x.to_numpy()

        return np.dot(test_x, self.partial_derivative_m)


def main ():
    pass

