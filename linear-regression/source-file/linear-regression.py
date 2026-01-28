from typing import Union
import numpy as np
import pandas as pd
from validator.DatasetValidation import DatasetValidation

# Importing for testing purposes
# Will be removed in the lin-reg-source.py
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
 
class LinearRegression:
    """Linear regression trained with (batch) gradient descent.

    Learns a linear mapping from features to a continuous target.

    The prediction function is:
    `y_hat = Xw + b` (when `fit_intercept=True`).

    Parameters
    ----------
    epoch : int
        Number of training iterations (epochs).
    learning_rate : float, default=1e-4
        Step size used for gradient descent updates.
    fit_intercept : bool, default=True
        Whether to learn an intercept/bias term `b`.

    Attributes
    ----------
    partial_derivative_m : np.ndarray or None
        Weight vector `w`. Initialized in `fit` with shape `(n_features,)`.
    partial_derivative_b : float or None
        Bias term `b`. Initialized in `fit` when `fit_intercept=True`.
    epochs : int
        Number of training iterations.
    learning_rate : float
        Step size for gradient updates.
    fit_intercept : bool
        Whether the model uses an intercept term.
    validator : DatasetValidation
        Dataset validation helper.

    Notes
    -----
    - This implementation uses batch gradient descent.
    - The training loop prints per-epoch information.
    """

    def __init__ (self, epoch: int, learning_rate: float = 1e-4, fit_intercept: bool = True):
        """Initialize the LinearRegression model.

        Parameters
        ----------
        epoch : int
            Number of training iterations (epochs).
        learning_rate : float, default=1e-4
            Step size for gradient descent updates.
        fit_intercept : bool, default=True
            Whether to learn an intercept/bias term.
        """
        self.partial_derivative_m = None
        self.partial_derivative_b = None
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.validator = DatasetValidation()

    def _initialize_weights_bias (self, train_x: np.ndarray):
        """Initialize model weights (and bias if enabled).

        Parameters
        ----------
        train_x : np.ndarray
            Training feature matrix with shape `(n_samples, n_features)`.

        Returns
        -------
        None
        """
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            self.partial_derivative_b = 0.0

    def _compute_cost (self, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute the training loss.

        Notes
        -----
        Despite the name, this returns the sum of squared errors (SSE), not
        the mean squared error (MSE).

        Parameters
        ----------
        train_y : np.ndarray
            Ground-truth targets.
        pred_y : np.ndarray
            Model predictions.

        Returns
        -------
        float
            Sum of squared errors over all samples.
        """
        return np.sum(np.square(pred_y - train_y))

    def _compute_weights_gradient (self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute gradients for the weights.

        Uses the batch gradient for squared error:
        `dw = (1 / m) * X^T (y_pred - y)`.

        Parameters
        ----------
        train_x : np.ndarray
            Feature matrix with shape `(m, n_features)`.
        train_y : np.ndarray
            Targets with shape `(m,)` or `(m, 1)`.
        pred_y : np.ndarray
            Predictions with shape compatible with `train_y`.

        Returns
        -------
        np.ndarray
            Weight gradients with shape `(n_features,)`.
        """
        return 1 / len(train_x) * np.dot(train_x.T, (pred_y - train_y))

    def _compute_bias_gradients (self, train_y: np.ndarray, pred_y: np.ndarray):
        """Compute gradient for the bias term.

        Uses:
        `db = (1 / m) * sum(y_pred - y)`.

        Parameters
        ----------
        train_y : np.ndarray
            Targets with shape `(m,)` or `(m, 1)`.
        pred_y : np.ndarray
            Predictions with shape compatible with `train_y`.

        Returns
        -------
        float
            Bias gradient.
        """
        return 1 / len(train_y) * np.sum(pred_y - train_y)

    def _update_weights_gradient (self, computed_weights_gradients: np.ndarray):
        """Apply a gradient update step to the weights.

        Update rule:
        `w = w - learning_rate * dw`.

        Parameters
        ----------
        computed_weights_gradients : np.ndarray
            Weight gradients `dw`.

        Returns
        -------
        None
        """
        self.partial_derivative_m -= self.learning_rate * computed_weights_gradients

    def _update_bias_gradient (self, computed_bias_gradients: np.ndarray):
        """Apply a gradient update step to the bias.

        Update rule:
        `b = b - learning_rate * db`.

        Parameters
        ----------
        computed_bias_gradients : float
            Bias gradient `db`.

        Returns
        -------
        None
        """
        self.partial_derivative_b -= self.learning_rate * computed_bias_gradients

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame],
    ):
        """Fit the model to training data.

        Parameters
        ----------
        train_x : np.ndarray or pandas.DataFrame
            Training features with shape `(n_samples, n_features)`.
        train_y : np.ndarray or pandas.DataFrame
            Training targets with shape `(n_samples,)` or `(n_samples, 1)`.

        Returns
        -------
        None

        Notes
        -----
        The training loop computes predictions as:
        `y_hat = Xw + b`.
        """
        if isinstance(train_x, pd.DataFrame):
            train_x = train_x.to_numpy()
            print(f"[+] Train x is Pandas. Converted to Numpy -> Rows: {train_x.shape[0]} | Columns: {train_x.shape[1]}")
        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.to_numpy()
            print(f"[+] Train y is Pandas. Converted to Numpy -> Rows: {train_y.shape[0]} | Columns: {train_y.shape[1]}")

        if self.validator.perform_dataset_validation(train_x, train_y):
            print("Validator checks passed. Proceeding with dataset checking")

        self._initialize_weights_bias(train_x)
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1} | M: {self.partial_derivative_m} | B: {self.partial_derivative_b}")
            
            # Main training loop
            predictions = np.dot(train_x, self.partial_derivative_m) + self.partial_derivative_b
            computed_weights = self._compute_weights_gradient(train_x, train_y, predictions)
            computed_bias = self._compute_bias_gradients(train_y, predictions)
            self._update_weights_gradient(computed_weights)
            self._update_bias_gradient(computed_bias)

    def predict (self, test_x: Union[np.ndarray | pd.DataFrame]):
        """Predict targets for the given samples.

        Computes `Xw + b` using the learned parameters.

        Parameters
        ----------
        test_x : np.ndarray or pandas.DataFrame
            Feature matrix with shape `(n_samples, n_features)`.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if isinstance(test_x, pd.DataFrame):
            print(f"Test x is Pandas. Converted to Numpy -> Rows: {test_x.shape[0]} | Columns: {test_x.shape[1]}")
            test_x.to_numpy()

        if self.validator.perform_dataset_validation(test_x, None):
            pass

        return np.dot(test_x, self.partial_derivative_m) + self.partial_derivative_b

# TESTING PURPOSES ONLY!
def main ():
    # NOTE: `main()` is for testing/benchmarking only.
    import time
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression

    linreg_instance = LinearRegression(epoch=2700, learning_rate=1e-3)
    X, Y = make_regression(
        n_samples=1_000_000,
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

    # Silence the per-epoch prints during benchmarking.
    t0 = time.perf_counter()
    linreg_instance.fit(tr_x, tr_y)
    t1 = time.perf_counter()

    preds = linreg_instance.predict(ts_x)
    mse = mean_squared_error(ts_y, preds)
    mae = mean_absolute_error(ts_y, preds)
    rmse = float(np.sqrt(mse))
    y_mean = float(np.mean(ts_y))
    y_std = float(np.std(ts_y))
    y_min = float(np.min(ts_y))
    y_max = float(np.max(ts_y))

    print(f"[+] Target (test) mean/std: {y_mean:.6f} / {y_std:.6f} | min/max: {y_min:.6f} / {y_max:.6f}")
    print(f"[+] Custom LinReg fit seconds: {t1 - t0:.3f}")
    print(f"[+] Custom LinReg Mean Squared Error: {mse}")
    print(f"[+] Custom LinReg Root Mean Squared Error: {rmse}")
    print(f"[+] Custom LinReg Mean Absolute Error: {mae}")
    if y_std != 0.0:
        print(f"[+] Custom LinReg MAE/std(y): {100 * (mae / y_std):.2f}%")
        print(f"[+] Custom LinReg RMSE/std(y): {100 * (rmse / y_std):.2f}%")
    print(f"[+] Custom LinReg R2 Score: {r2_score(ts_y, preds)}")

    skl = SklearnLinearRegression(fit_intercept=True)
    t2 = time.perf_counter()
    skl.fit(tr_x, tr_y)
    t3 = time.perf_counter()
    skl_preds = skl.predict(ts_x)
    print(f"\n[+] Sklearn LinReg fit seconds: {t3 - t2:.3f}")
    print(f"[+] Sklearn LinReg Mean Squared Error: {mean_squared_error(ts_y, skl_preds)}")
    print(f"[+] Sklearn LinReg Mean Absolute Error: {mean_absolute_error(ts_y, skl_preds)}")
    print(f"[+] Sklearn LinReg R2 Score: {r2_score(ts_y, skl_preds)}")

main()

