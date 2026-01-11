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
    """Binary logistic regression trained with (batch) gradient descent.

    The model computes linear scores `z = Xw + b`, converts them to probabilities
    via the sigmoid function, and (in `predict`) thresholds probabilities at 0.5
    to produce class labels in {0, 1}.

    Attributes:
        partial_derivative_m (np.ndarray | None): Weight vector (initialized in `fit`).
        partial_derivative_b (float | None): Bias term (initialized in `fit` if `fit_intercept=True`).
        epochs (int): Number of training iterations.
        learning_rate (int | float): Step size for gradient updates.
        fit_intercept (bool): Whether to learn an intercept/bias term.
        _dset_validator (DatasetValidation): Dataset validation helper.
        _hyperparameter_validator (ParameterValidator): Hyperparameter validation helper.
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

    def _initialize_weights_bias(self, train_x: np.ndarray):
        """Initialize model parameters.

        Args:
            train_x (np.ndarray): Training feature matrix of shape (n_samples, n_features).

        Returns:
            None
        """
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            self.partial_derivative_b = 0.0

    def _compute_weights_gradients(self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute gradients for the weights.

        Uses the batch gradient:
        `dw = (1 / m) * X^T (y_pred - y)`.

        Args:
            train_x (np.ndarray): Feature matrix of shape (m, n_features).
            train_y (np.ndarray): Targets of shape (m,) or (m, 1).
            pred_y (np.ndarray): Predicted probabilities of shape compatible with `train_y`.

        Returns:
            np.ndarray: Weight gradients of shape (n_features,).
        """
        return 1 / len(train_x) * np.dot(train_x.T, (pred_y - train_y))

    def _compute_bias_gradients(self, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute gradient for the bias term.

        Uses:
        `db = (1 / m) * sum(y_pred - y)`.

        Args:
            train_y (np.ndarray): Targets of shape (m,) or (m, 1).
            pred_y (np.ndarray): Predicted probabilities of shape compatible with `train_y`.

        Returns:
            float: Bias gradient.
        """
        return 1 / len(train_y) * np.sum(pred_y - train_y)

    def _update_weights(self, computed_weights_gradients: np.ndarray):
        """Apply a gradient update step to the weights.

        Update rule:
        `w = w - learning_rate * dw`.

        Args:
            computed_weights_gradients (np.ndarray): Weight gradients (dw).

        Returns:
            None
        """
        self.partial_derivative_m -= self.learning_rate * computed_weights_gradients

    def _update_bias(self, computed_bias_gradient: float):
        """Apply a gradient update step to the bias.

        Update rule:
        `b = b - learning_rate * db`.

        Args:
            computed_bias_gradient (float): Bias gradient (db).

        Returns:
            None
        """
        self.partial_derivative_b -= self.learning_rate * computed_bias_gradient

    def _sigmoid_function(self, predictions: np.ndarray):
        """Apply the sigmoid function to convert scores to probabilities.

        Computes: `sigma(z) = 1 / (1 + exp(-z))`.

        Args:
            predictions (np.ndarray): Raw linear scores/logits.

        Returns:
            np.ndarray: Probabilities in the range (0, 1).
        """
        return 1 / (1 + np.exp(-predictions))

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame]
    ):
        """Train the model on the provided dataset.

        Args:
            train_x (np.ndarray | pd.DataFrame): Training features of shape (n_samples, n_features).
            train_y (np.ndarray | pd.DataFrame): Binary targets of shape (n_samples,) or (n_samples, 1).

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

    def predict(self, test_x: Union[np.ndarray | pd.DataFrame]):
        """Predict class labels for the given samples.

        Produces probabilities with sigmoid and thresholds at 0.5.

        Args:
            test_x (np.ndarray | pd.DataFrame): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
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

