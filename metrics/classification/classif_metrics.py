from typing import Union, Callable
import numpy as np
import pandas as pd
from validator.validator import DatasetValidation, ClassificationMetricValidation

dset_val = DatasetValidation()
classif_val = ClassificationMetricValidation()

"""
TODO:
    -- Write logic for binary classification
        --- Write comparing logic for TP, TN, FP, FN
    -- Write logic for multiclass classification
        --- Write comparing logic for multiclass TP, TN, FP, FN

P.S Isn't binary and multiclass comparison logic the same?

REMOVE IN DEPLOYMENT!
"""
def metric_preds_counter (
    true_y: np.ndarray,
    pred_y: np.ndarray,
    prediction_type: str = "binary"
):
    if prediction_type == "binary":
        true_positive = np.sum(np.logical_and(true_y == 1, pred_y == 1))
        true_negative = np.sum(np.logical_and(true_y == 0, pred_y == 0))
        false_positive = np.sum(np.logical_and(true_y == 0, pred_y == 1))
        false_negative = np.sum(np.logical_and(true_y == 1, pred_y == 0))

        return (
            true_positive,
            true_negative,
            false_positive,
            false_negative
        )

    if prediction_type == "multiclass":
        pass

def metric_validator_helper (
    true_y: np.ndarray,
    pred_y: np.ndarray, 
    classif_metric: Callable
):
    dataset_validated = False
    classif_metric_validated = False

    if (dset_val.validate_existence([true_y, pred_y]) and
        dset_val.validate_shapes(true_y, pred_y) and
        dset_val.validate_types(true_y, pred_y)
    ):
        dataset_validated = True

    if (classif_val.validate_zero_division(true_y) and
        classif_val.validate_label_count(true_y, classif_metric)
    ):
        classif_metric_validated = True

    if dataset_validated and classif_metric_validated:
        return True

def accuracy (
    true_y: Union[np.ndarray | pd.DataFrame], 
    pred_y: Union[np.ndarray | pd.DataFrame],
    average="binary"
):
    if metric_validator_helper(true_y, pred_y, accuracy):
        true_positive, true_negative, _, _ = metric_preds_counter(true_y, pred_y)
        return (len(true_positive) + len(true_negative)) / len(pred_y)

def precision (
    true_y: Union[np.ndarray | pd.DataFrame], 
    pred_y: Union[np.ndarray | pd.DataFrame]
):
    if metric_validator_helper(true_y, pred_y, precision):
        true_positive, _, false_positive, _ = metric_preds_counter(true_y, pred_y)
        return len(true_positive) / (len(true_positive) + len(false_positive))

def recall (
    true_y: Union[np.ndarray | pd.DataFrame], 
    pred_y: Union[np.ndarray | pd.DataFrame]
):
    if metric_validator_helper(true_y, pred_y, precision):
        true_positive, _, _, false_negative = metric_preds_counter(true_y, pred_y)
        return len(true_positive) / (len(true_positive) + len(false_negative))

def f1 (
    true_y: Union[np.ndarray | pd.DataFrame], 
    pred_y: Union[np.ndarray | pd.DataFrame]
):
    if metric_validator_helper(true_y, pred_y, f1):
        precision_score = precision(true_y, pred_y)
        recall_score = recall(true_y, pred_y)
        return 2 * ((precision_score * recall_score) / (precision_score + recall_score))

def log_loss (
    true_y: Union[np.ndarray | pd.DataFrame], 
    pred_y: Union[np.ndarray | pd.DataFrame]
):
    if metric_validator_helper(true_y, pred_y, f1):
        true_y_difference = 1 - true_y
        pred_y_difference = 1 - pred_y
        return -1 / len(true_y) * np.sum(true_y * np.log(pred_y) + true_y_difference * pred_y_difference)

