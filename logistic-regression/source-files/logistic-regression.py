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

    The model computes linear scores ``z = Xw + b``, converts them to
    probabilities via the sigmoid function, and (in ``predict``) thresholds
    probabilities at 0.5 to produce class labels in {0, 1}.

    Parameters
    ----------
    epochs : int
        Number of training iterations (complete passes over the dataset).
    learning_rate : int or float
        Step size for gradient updates.
    fit_intercept : bool, default=True
        Whether to learn an intercept/bias term.

    Attributes
    ----------
    partial_derivative_m : np.ndarray or None
        Weight vector of shape (n_features,). Initialized in ``fit``.
    partial_derivative_b : float or None
        Bias term. Initialized in ``fit`` if ``fit_intercept=True``.
    epochs : int
        Number of training iterations.
    learning_rate : int or float
        Step size for gradient updates.
    fit_intercept : bool
        Whether to learn an intercept/bias term.
    _dset_validator : DatasetValidation
        Dataset validation helper.
    _hyperparameter_validator : ParameterValidator
        Hyperparameter validation helper.
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
        self.fit_intercept = fit_intercept
        self._dset_validator = DatasetValidation()
        self._hyperparameter_validator = ParameterValidator()

    def _initialize_weights_bias(self, train_x: np.ndarray):
        """Initialize model parameters.

        Sets the weight vector to zeros and, if ``fit_intercept`` is True,
        sets the bias to 0.0.

        Parameters
        ----------
        train_x : np.ndarray of shape (n_samples, n_features)
            Training feature matrix used to determine the number of weights.
        """
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            self.partial_derivative_b = 0.0

    def _compute_weights_gradients(self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute gradients for the weights.

        Uses the batch gradient:
        `dw = (1 / m) * X^T (y_pred - y)`.

        Parameters
        ----------
        train_x : np.ndarray of shape (m, n_features)
            Feature matrix.
        train_y : np.ndarray of shape (m,) or (m, 1)
            True binary target values.
        pred_y : np.ndarray of shape (m,) or (m, 1)
            Predicted probabilities, same shape as `train_y`.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Weight gradients.
        """
        return 1 / len(train_x) * np.dot(train_x.T, (pred_y - train_y))

    def _compute_bias_gradients(self, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute gradient for the bias term.

        Uses the batch gradient:

        .. math::

            db = \\frac{1}{m} \\sum (\\hat{y} - y)

        Parameters
        ----------
        train_y : np.ndarray of shape (m,) or (m, 1)
            True binary target values.
        pred_y : np.ndarray of shape (m,) or (m, 1)
            Predicted probabilities, same shape as `train_y`.

        Returns
        -------
        float
            Bias gradient scalar.
        """
        return 1 / len(train_y) * np.sum(pred_y - train_y)

    def _update_weights(self, computed_weights_gradients: np.ndarray):
        """Apply a gradient update step to the weights.

        Update rule:
        `w = w - learning_rate * dw`.

        Parameters
        ----------
        computed_weights_gradients : np.ndarray of shape (n_features,)
            Weight gradients (dw).
        """
        self.partial_derivative_m -= self.learning_rate * computed_weights_gradients

    def _update_bias(self, computed_bias_gradient: float):
        """Apply a gradient update step to the bias.

        Update rule:
        `b = b - learning_rate * db`.

        Parameters
        ----------
        computed_bias_gradient : float
            Bias gradient (db).
        """
        self.partial_derivative_b -= self.learning_rate * computed_bias_gradient

    def _sigmoid_function(self, predictions: np.ndarray):
        """Apply the sigmoid function to convert scores to probabilities.

        Computes: `sigma(z) = 1 / (1 + exp(-z))`.

        Parameters
        ----------
        predictions : np.ndarray
            Raw linear scores (logits).

        Returns
        -------
        np.ndarray
            Probabilities in the range (0, 1).
        """
        return 1 / (1 + np.exp(-predictions))

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame]
    ):
        """Train the model on the provided dataset.

        Performs batch gradient descent for ``epochs`` iterations. At each
        epoch the method computes the sigmoid predictions
        ``z = Xw + b``, derives the weight and bias gradients, and updates
        the parameters in-place.

        If the inputs are ``pd.DataFrame`` objects they are converted to
        NumPy arrays before training.

        Parameters
        ----------
        train_x : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Training feature matrix.
        train_y : np.ndarray or pd.DataFrame of shape (n_samples,) or (n_samples, 1)
            Binary target values.
        """
        if isinstance(train_x, pd.DataFrame):
            train_x = train_x.to_numpy()
            print(f"[+] Train x is Pandas, Converted to Numpy -> Rows: {train_x.shape[0]} | Columns: {train_x.shape[1]}")
        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.to_numpy()
            print(f"[+] Train y is Pandas, Converted to Numpy -> Rows: {train_y.shape[0]} | Columns: {train_y.shape[1]}")

        self._dset_validator.perform_dataset_validation(train_x, train_y)
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

        Computes raw logits ``z = Xw + b``, passes them through the sigmoid
        function, and thresholds the resulting probabilities at 0.5 to
        produce class labels in {0, 1}.

        If the input is a ``pd.DataFrame`` it is converted to a NumPy array
        before prediction.

        Parameters
        ----------
        test_x : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
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

