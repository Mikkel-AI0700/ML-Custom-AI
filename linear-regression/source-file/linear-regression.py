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
    """
    Supervised machine learning model that models the relationship of the
    independent and dependent variables

    Parameters:
        epochs (int): The amount of iterations that the dataset will go through the model
        learning_rate (Union[int | float): Controls how much the gradients will be updated. Controls gradient step
        fit_intercept (bool): Controls whether to include the bias. If no, then model will assume data goes through (0, 0) 
        
    Returns:
        LinearRegression (object): The instantiated LinearRegression object
    """
    def __init__ (self, epoch: int, learning_rate: float = 1e-4, fit_intercept: bool = True):
        self.partial_derivative_m = None
        self.partial_derivative_b = None
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.validator = DatasetValidation()

    def _initialize_weights_bias (self, train_x: np.ndarray):
        """
        Initializes the weights and bias of the linear regression model

        Parameters: 
            train_x (np.ndarray): The main training dataset to be used for training

        Returns:
            None
        """
        self.partial_derivative_m = np.zeros((train_x.shape[1]))

        if self.fit_intercept:
            self.partial_derivative_b = 0.0

    def _compute_cost (self, train_y: np.ndarray, pred_y: np.ndarray):
        """
        Computing the Mean Squared Error (MSE) error of the linear regression model

        Parameters:
            train_y (np.ndarray): The training ground truths of the training dataset
            pred_y (np.ndarray): The initial predictions of the model during training

        Returns:
            computed_mse (np.float32): The computed MSE error for the model
        """
        return np.sum(np.square(pred_y - train_y))

    def _compute_weights_gradient (self, train_x: np.ndarray, train_y: np.ndarray, pred_y: np.ndarray):
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

    def _update_weights_gradient (self, computed_weights_gradients: np.ndarray):
        """
        Updates the model's weights using: w = w - (learning_rate * dw)

        Parameters:
            computed_weights_gradients (np.ndarray): A ndarray containing the computed gradients from _compute_weights_gradients

        Returns:
            None
        """
        self.partial_derivative_m -= self.learning_rate * computed_weights_gradients

    def _update_bias_gradient (self, computed_bias_gradients: np.ndarray):
        """
        Updates the model's bias using: b = b - (learning_rate * db)

        Parameters:
            computed_bias_gradient (float): A single float value containing the computed bias from _compute_bias_gradients

        Returns:
            None
        """
        self.partial_derivative_b -= self.learning_rate * computed_bias_gradients

    def fit (
        self,
        train_x: Union[np.ndarray | pd.DataFrame],
        train_y: Union[np.ndarray | pd.DataFrame],
    ):
        """
        Starts training the logistic regression model

        Parameters:
            train_x: (Union[np.ndarray | pd.DataFrame]): The main training dataset
            train_y: (Union[np.ndarray | pd.DataFrame]): The main ground truths dataset

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
        """
        Taking in a np.ndarray of test samples then inferencing them using: Xw + b

        Parameters:
            test_x (Union[np.ndarray | pd.DataFrame]): The dataset that will be inferenced

        Returns:
            predictions (np.ndarray): The inferenced elements
        """
        if self.validator.validate_existence([test_x]):
            pass
        if isinstance(test_x, pd.DataFrame):
            print("Test x is Pandas. Converted to Numpy -> Rows: {train_x.shape[0]} | Columns: {train_x.shape[1]}")
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

