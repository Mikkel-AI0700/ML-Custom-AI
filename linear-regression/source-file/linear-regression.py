from typing import Union
import numpy as np
import pandas as pd
from validator.validator import DatasetValidation

# Importing for testing purposes
# Will be removed in the lin-reg-source.py
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
 
class LinearRegression:
    """Linear regression trained with (batch) gradient descent.

    This model learns a linear mapping from features to a continuous target:
    `y_hat = Xw + b` (if `fit_intercept=True`).

    Attributes:
        partial_derivative_m (np.ndarray | None): Weight vector (initialized in `fit`).
        partial_derivative_b (float | None): Bias term (initialized in `fit` if `fit_intercept=True`).
        epochs (int): Number of training iterations.
        learning_rate (float): Step size for gradient updates.
        fit_intercept (bool): Whether to learn an intercept term.
        validator (DatasetValidation): Dataset validation helper.
    """

    def __init__ (self, epoch: int, learning_rate: float = 1e-4, fit_intercept: bool = True):
        """Initialize the LinearRegression model.

        Args:
            epoch (int): Number of training iterations.
            learning_rate (float, optional): Step size for gradient updates. Defaults to 1e-4.
            fit_intercept (bool, optional): Whether to learn an intercept/bias term. Defaults to True.
        """
        self.partial_derivative_m = None
        self.partial_derivative_b = None
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.validator = DatasetValidation()

    def _initialize_weights_bias (self, train_x: np.ndarray):
        """Initialize model weights (and bias if enabled).

        Args:
            train_x (np.ndarray): Training feature matrix of shape (n_samples, n_features).

        Returns:
            None
        """
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            self.partial_derivative_b = 0.0

    def _compute_cost (self, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute the training loss.

        Note:
            Despite the name, this returns the *sum* of squared errors (SSE),
            not the mean squared error (MSE).

        Args:
            train_y (np.ndarray): Ground-truth targets.
            pred_y (np.ndarray): Model predictions.

        Returns:
            float: Sum of squared errors over all samples.
        """
        return np.sum(np.square(pred_y - train_y))

    def _compute_weights_gradient (self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute gradients for the weights.

        Uses the batch gradient for squared error:
        `dw = (1 / m) * X^T (y_pred - y)`.

        Args:
            train_x (np.ndarray): Feature matrix of shape (m, n_features).
            train_y (np.ndarray): Targets of shape (m,) or (m, 1).
            pred_y (np.ndarray): Predictions of shape compatible with `train_y`.

        Returns:
            np.ndarray: Weight gradients of shape (n_features,).
        """
        return 1 / len(train_x) * np.dot(train_x.T, (pred_y - train_y))

    def _compute_bias_gradients (self, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute gradient for the bias term.

        Uses:
        `db = (1 / m) * sum(y_pred - y)`.

        Args:
            train_y (np.ndarray): Targets of shape (m,) or (m, 1).
            pred_y (np.ndarray): Predictions of shape compatible with `train_y`.

        Returns:
            float: Bias gradient.
        """
        return 1 / len(train_y) * np.sum(pred_y - train_y)

    def _update_weights_gradient (self, computed_weights_gradients: np.ndarray):
        """Apply a gradient update step to the weights.

        Update rule:
        `w = w - learning_rate * dw`.

        Args:
            computed_weights_gradients (np.ndarray): Weight gradients (dw).

        Returns:
            None
        """
        self.partial_derivative_m -= self.learning_rate * computed_weights_gradients

    def _update_bias_gradient (self, computed_bias_gradients: np.ndarray):
        """Apply a gradient update step to the bias.

        Update rule:
        `b = b - learning_rate * db`.

        Args:
            computed_bias_gradients (float): Bias gradient (db).

        Returns:
            None
        """
        self.partial_derivative_b -= self.learning_rate * computed_bias_gradients

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame],
    ):
        """Train the model on the provided dataset.

        Args:
            train_x (np.ndarray | pd.DataFrame): Training features of shape (n_samples, n_features).
            train_y (np.ndarray | pd.DataFrame): Training targets of shape (n_samples,) or (n_samples, 1).

        Returns:
            None
        """
        if (self.validator.validate_existence([train_x, train_y]) and
            self.validator.validate_shapes(train_x, train_y)
        ):
            print("Validator checks passed. Proceeding with dataset checking")

        if isinstance(train_x, pd.DataFrame):
            train_x = train_x.to_numpy()
            print(f"[+] Train x is Pandas. Converted to Numpy -> Rows: {train_x.shape[0]} | Columns: {train_x.shape[1]}")
        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.to_numpy()
            print(f"[+] Train y is Pandas. Converted to Numpy -> Rows: {train_y.shape[0]} | Columns: {train_y.shape[1]}")

        self._initialize_weights_bias(train_x)
        for epoch in range(self.epochs):
            print(f"Batch: {epoch + 1} | M: {self.partial_derivative_m}")
            
            # Main training loop
            predictions = np.dot(train_x, self.partial_derivative_m) + self.partial_derivative_b
            computed_weights = self._compute_weights_gradient(train_x, train_y, predictions)
            computed_bias = self._compute_bias_gradients(train_y, predictions)
            self._update_weights_gradient(computed_weights)
            self._update_bias_gradient(computed_bias) 

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        """Predict targets for the given samples.

        Computes `Xw + b` using the learned parameters.

        Args:
            test_x (np.ndarray | pd.DataFrame): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        if self.validator.validate_existence([test_x]):
            pass
        if isinstance(test_x, pd.DataFrame):
            print(f"Test x is Pandas. Converted to Numpy -> Rows: {test_x.shape[0]} | Columns: {test_x.shape[1]}")
            test_x.to_numpy()

        #test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])

        return np.dot(test_x, self.partial_derivative_m) + self.partial_derivative_b

# TESTING PURPOSES ONLY!
# main() FUNCTION WILL BE REMOVED IN lin-reg-source.py
def main ():
    linreg_instance = LinearRegression(epoch=2300, learning_rate=1e-3)
    X, Y = make_regression(
        n_samples=200_000,
        n_features=5,
        random_state=42
    )

    tr_x, ts_x, tr_y, ts_y = train_test_split(
        X,
        Y,
        train_size=0.7,
        test_size=0.3,
        shuffle=True,
        random_state=42
    )

    linreg_instance.fit(tr_x, tr_y)

    preds = linreg_instance.predict(ts_x)
    print(f"[+] MSE: {mean_squared_error(ts_y, preds)}")    

main()

