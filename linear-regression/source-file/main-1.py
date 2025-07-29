from typing import Union
import numpy as np
import pandas as pd
import scipy as scp

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

class MSE:
    def __init__ (self):
        self.NULL_DATASET_ERROR = "[-] Error: One of the datasets is null"
        self.TYPE_MISMATCH_ERROR = "[-] Error: Datasets don't have equal shapes"

    def _validate_predictions (
        self,
        test_preds: Union[np.ndarray | pd.DataFrame],
        model_preds: Union[np.ndarray | pd.DataFrame]   
    ):
        try:
            if test_preds is None or model_preds is None:
                raise TypeError(self.NULL_DATASET_ERROR)
            elif test_preds.shape != model_preds.shape:
                raise ValueError(self.NULL_DATASET_ERROR)
            else:
                return True
        except TypeError as null_dataset_error:
            print(null_dataset_error)
            exit(EXIT_FAILURE)
        except ValueError as shape_mismatch_error:
            print(shape_mismatch_error)
            exit(EXIT_FAILURE)

    def compute_loss (
        self,
        test_predictions: Union[np.ndarray | pd.DataFrame],
        model_predictions: Union[np.ndarray | pd.DataFrame]
    ):
        if self._validate_predictions(test_predictions, model_predictions):
            return np.mean(np.square(test_predictions - model_predictions))
        
class MSE:
    def __init__ (self):
        pass

    def compute (
        self,
        test_preds: Union[np.ndarray | pd.DataFrame],
        model_preds: Union[np.ndarray | pd.DataFrame]
    ):
        try:
            if test_preds != None and model_preds != None;
                pass
            else:
                raise ValueError("[-] Error: Either one or both of the datasets is null")
        except ValueError as null_dataset_error:
            print(null_dataset_error)
            exit(EXIT_FAILURE)

class LinearRegression:
    def __init__(self, num_of_epochs: int, learning_rate: float, fit_intercept: bool = True):
        self.partial_dev_m = None
        self.partial_dev_b = None
        self.epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept 

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
        try:
            if any(dataset == None for dataset in [train_x, train_y]):
                raise ValueError("[-] Error: One or both the datasets is null")
            if self.fit_intercept:
                pass

            for epoch in self.epochs:
                print(f"[+] Epoch: {epoch + 1}")
                predictions = self.partial_dev_m * train_x + self.partial_dev_b
                weight_gradient = self._compute_weights(train_x, train_y, predictions)
                bias_gradient = self._compute_bias(train_x, train_y, predictions)
                self._update_weights_gradients(weight_gradient)
                self._update_bias_gradients(bias_gradient)
        except ValueError as null_dataset_error:
            print(null_dataset_error)
            exit(EXIT_FAILURE)

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        return self.partial_dev_m * test_x + self.partial_dev_b