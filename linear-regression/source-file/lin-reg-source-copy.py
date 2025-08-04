# Python SL
from typing import Union

# Python third party imports
import numpy as np
import pandas as pd
import scipy as scp
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Python local imports
from validator.validators import Validate

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# Setting the global random seed pseudo number
np.random.seed(42)

class MeanSquaredError:
    def __init__ (self):
        self.validator = Validator()

    def compute (
        self,
        test_preds: Union[np.ndarray | pd.DataFrame],
        model_preds: Union[np.ndarray | pd.DataFrame]
    ):
        if self.validator.validate([test_preds, model_preds]):
            return 1 / float(len(test_preds)) * np.mean((test_preds - model_preds) ** 2)

class LinearRegression:
    def __init__(self, num_of_epochs: int = None, learning_rate: float = None, fit_intercept: bool = True):
        self.partial_dev_m = None
        self.partial_dev_b = None
        self.epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.validator = Validator()

    def _compute_weights (self, train_x: np.ndarray, train_y: np.ndarray, predictions: np.ndarray):
        return -2 / len(train_x) * np.sum(train_x * (train_y - predictions))

    def _compute_bias (self, train_x: np.ndarray, train_y: np.ndarray, predictions: np.ndarray):
        return -2 / len(train_x) * np.sum(train_y - predictions)

    def _update_weights_gradients (self, computed_weight_gradient: Union[int | float]):
        self.partial_dev_m -= self.learning_rate * computed_weight_gradient

    def _update_bias_gradients (self, computed_bias_gradient: Union[int | float]):
        self.partial_dev_b -= self.learning_rate * computed_bias_gradient

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame]
    ):
        # Initializing the weights and "bias"
        train_x = np.hstack([np.ones(train_x.shape[0], 1), train_x])
        self.partial_dev_m = np.zeros((train_x.shape[1]))

        if self.validator.validate([train_x, train_y]):
            for epoch in range(self.epochs):
                print(f"[+] Epoch: {epoch} | Partial Dev M: {self.partial_dev_m} | Partial Dev B: {self.partial_dev_b}\n")
                predictions = np.dot(train_x, self.partial_dev_m)
                weights_gradient = self._compute_weights(train_x, train_y, predictions)
                bias_gradient = self._compute_bias(train_x, train_y, predictions)
                self._update_weights_gradients(weights_gradient)
                self._update_bias_gradients(bias_gradient)

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        if self.validator.validate([test_x]):
            return self.partial_dev_m * test_x + self.partial_dev_b

def main ():
    # Loading the datasets
    diabetes_X, diabetes_Y = load_diabetes(return_X_y=True)
    tr_x, ts_x, tr_y, ts_y = train_test_split(
        diabetes_X,
        diabetes_Y,
        train_size=0.7,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    lin_reg = LinearRegression()
    lin_reg.fit(tr_x, tr_y)

main()

