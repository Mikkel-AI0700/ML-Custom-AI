from typing import Union
import numpy as np
import pandas as pd
import scipy as scp

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# Setting the global random seed pseudo number
np.random.seed(42)

class Validator:
    def __init__ (self):
        self.NULL_DATASET_ERROR = "[-] Error: Either or both datasets is null"
        self.UNEQUAL_SHAPE_ERROR = "[-] Error: Shapes of both datasets aren't equal"

    def validate (self, dataset: list[np.ndarray | pd.DataFrame]):
        try:
            if any(dset.shape == None for dset in dataset):
                raise TypeError(self.NULL_DATASET_ERROR)
            if len(dataset) == 2:
                if dataset[0].shape[0] != dataset[1].shape[0]:
                    raise ValueError(self.UNEQUAL_SHAPE_ERROR)

            return True
        except TypeError as null_dataset_error:
            print(null_dataset_error)
            exit(EXIT_FAILURE)
        except ValueError as shape_mismatch_error:
            print(shape_mismatch_error)
            exit(EXIT_FAILURE)

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
        self.partial_dev_m = np.random.rand()
        self.partial_dev_b = np.random.rand() if fit_intercept else 0.0
        self.epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.validator = Validator()

    def _compute_weights (self, train_x: np.ndarray, train_y: np.ndarray, predictions: np.ndarray):
        return -(2 / float(len(train_x))) * np.sum(train_x * (train_y - predictions))

    def _compute_bias (self, train_x: np.ndarray, train_y: np.ndarray, predictions: np.ndarray):
        return -(2 / float(len(train_x))) * np.sum(train_y - predictions)

    def _update_weights_gradients (self, computed_weight_gradient: Union[int | float]):
        self.partial_dev_m -= self.learning_rate * computed_weight_gradient

    def _update_bias_gradients (self, computed_bias_gradient: Union[int | float]):
        self.partial_dev_b -= self.learning_rate * computed_bias_gradient

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame]
    ):
        if self.fit_intercept:
            train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])

        if self.validator.validate([train_x, train_y]):
            for epoch in range(self.epochs):
                print(f"[+] Epoch: {epoch} | Partial Dev M: {self.partial_dev_m} | Partial Dev B: {self.partial_dev_b}\n")
                predictions = self.partial_dev_m * train_x + self.partial_dev_b
                weights_gradient = self._compute_weights(train_x, train_y, predictions)
                bias_gradient = self._compute_bias(train_x, train_y, predictions)
                self._update_weights_gradients(weights_gradient)
                self._update_bias_gradients(bias_gradient)

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        if self.validator.validate([test_x]):
            return self.partial_dev_m * test_x + self.partial_dev_b

