from typing import Union
import numpy as np
import pandas as pd
from validator.DatasetValidation import DatasetValidation
from validator.ParameterValidator import ParameterValidator
from base.BaseEstimator import BaseEstimator
from base.ClassifierMixin import ClassifierMixin

# Importing scikit-learn classification metrics
# It will be removed in the main source files
from sklearn.metrics import accuracy_score, precision_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LogisticRegression:
    """
    Machine learning algorithm uses the sigmoid function to turn the logits
    from the initial Xw + b formula into probabilities

    Parameters:
        epochs (int): The amount of iterations that the dataset will go through the model
        learning_rate (Union[int | float): Controls how much the gradients will be updated. Controls gradient step
        fit_intercept (bool): Controls whether to include the bias. If no, then model will assume data goes through (0, 0)

    Returns:
        LogisticRegression (object): The instantiated LogisticRegression
    """
    __parameter_constraints__ = {
        "epochs": (int),
        "learning_rate": (int, float),
        "fit_intercept": (int)
    }

    def __init__ (self, epochs: int, learning_rate: Union[int | float], fit_intercept: bool = True):
        self.partial_derivative_m = None
        self.partial_derivative_b = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.fit_intercept = True
        self._dset_validator = DatasetValidation()
        self._hyperparameter_validator = ParameterValidator()
        self.validator = DatasetValidation()

    def _initialize_weights_bias (self, train_x: np.ndarray):
        """
        Initializes the weights and bias by counting the features present in
        the dataset and checking if the fit_intercept hyperparameter is set to true

        Parameters:
            train_x (np.ndarray): The main dataset to be used for training

        Returns:
            None
        """
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            self.partial_derivative_b = 0.0

    def _compute_weights_gradients (self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
        """
        Computes the required gradients to adjust model's weights using: dw = (1 / m) * sum((y_pred - y) * x)

        Parameters:
            train_x (np.ndarray): The main training dataset
            train_y (np.ndarray: The ground truths dataset
            pred_y (np.ndarray): The ndarray of logits after Xw + b

        Returns:
            computed_weights_gradients (np.ndarray): A ndarray of computed gradients to update model weights
        """
        return 1 / len(train_x) * np.dot(train_x.T, (pred_y - train_y))

    def _compute_bias_gradients (self, train_y: np.ndarray, pred_y: np.ndarray):
        """
        Computes the required bias to adjust the model's bias using: db = (1 / m) * sum(y_pred - y)

        Parameters:
            train_y (np.ndarray): The ground truths dataset
            pred_y (np.ndarray): The ndarray of logits after Xw + b

        Returns:
            computed_bias_gradient (np.float32): A np.float32 of computed bias to update the model's bias
        """
        return 1 / len(train_y) * np.sum(pred_y - train_y)

    def _update_weights (self, computed_weights_gradients: np.ndarray):
        """
        Updates the model's weights using: w = w - (learning_rate * dw)

        Parameters:
            computed_weights_gradients (np.ndarray): A ndarray containing the computed gradients from _compute_weights_gradients

        Returns:
            None
        """
        self.partial_derivative_m -= self.learning_rate * computed_weights_gradients

    def _update_bias (self, computed_bias_gradient: float):
        """
        Updates the model's bias using: b = b - (learning_rate * db)

        Parameters:
            computed_bias_gradient (float): A single float value containing the computed bias from _compute_bias_gradients

        Returns:
            None
        """
        self.partial_derivative_b -= self.learning_rate * computed_bias_gradient

    def _sigmoid_function (self, predictions: np.ndarray):
        """
        Converts the raw linear regression predictions into logits: 1 / (1 + e^-z)

        Parameters:
            predictions (np.ndarray): Contains the raw linear regression predictions to be converted.

        Returns:
            sigmoid_logits (np.ndarray): Converted predictions using sigmoid function.
        """
        return 1 / (1 + np.exp(-predictions))

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame]
    ):
        """
        Starts training the logistic regression model

        Parameters:
            train_x: (Union[np.ndarray | pd.DataFrame]): The main training dataset
            train_y: (Union[np.ndarray | pd.DataFrame]): The main ground truths dataset

        Returns:
            None
        """
        if isinstance(train_x, pd.DataFrame):
            train_x = train_x.to_numpy()
            print(f"[+] Train x is Pandas, Converted to Numpy -> Rows: {train_x.shape[0]} | Columns: {train_x.shape[1]}")
        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.to_numpy()
            print(f"[+] Train y is Pandas, Converted to Numpy -> Rows: {train_y.shape[0]} | Columns: {train_y.shape[1]}")

        self._dset_validator.perform_dataset_validation(train_x, train_y)
        self._hyperparameter_validator.validate_parameters()
        self._initialize_weights_bias(train_x)

        for epoch in range(self.epochs):
            print(f"[+] Epoch: {epoch + 1} | Partial derivative M: {self.partial_derivative_m}")

            # Main prediction loop
            sigmoid_predictions = self._sigmoid_function(np.dot(train_x, self.partial_derivative_m) + self.partial_derivative_b)

            # Main gradient computation and updating
            computed_weights = self._compute_weights_gradients(train_x, train_y, sigmoid_predictions)
            computed_bias = self._compute_bias_gradients(train_y, sigmoid_predictions)
            self._update_weights(computed_weights)
            self._update_bias(computed_bias)

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        """
        Taking in a np.ndarray of test samples then inferencing them using: Xw + b

        Parameters:
            test_x (Union[np.ndarray | pd.DataFrame]): The dataset that will be inferenced

        Returns:
            predictions (np.ndarray): The inferenced elements
        """
        if isinstance(test_x, pd.DataFrame):
            test_x = test_x.to_numpy()
            print(f"[+] Train x is Pandas, Converted to Numpy -> Rows: {test_x.shape[0]} | Columns: {test_x.shape[1]}")
        if self._dset_validator.perform_dataset_validation(test_x):
            pass

        logit_predictions = np.dot(test_x, self.partial_derivative_m) + self.partial_derivative_b
        squashed_predictions = self._sigmoid_function(logit_predictions)
        return np.where(squashed_predictions >= 0.5, 1, 0)

def main ():
    classification_generator_parameters = {
        "n_samples": 900_000,
        "n_features": 5,
        "n_informative": 3,
        "random_state": 42
    }

    logreg_instance_parameters = {
        "epochs": 1500,
        "learning_rate": 1e-2
    }

    logreg_instance = LogisticRegression(**logreg_instance_parameters)
    X, Y = make_classification(**classification_generator_parameters)

    tr_x, ts_x, tr_y, ts_y = train_test_split(
        X,
        Y,
        train_size = 0.8,
        test_size = 0.2,
        shuffle = True,
        random_state = 42
    )

    logreg_instance.fit(tr_x, tr_y)
    predictions = logreg_instance.predict(ts_x)

    print(f"Accuracy score: {accuracy_score(ts_y, predictions)} \nPrecision score: {precision_score(ts_y, predictions)}")

main()

