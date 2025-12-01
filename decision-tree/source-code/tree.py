from typing import Any, Union
import numpy as np
import pandas as pd
from base.EstimatorClass import BaseEstimator, ClassifierMixin

class Node:
    def __init__ (
        self,
        feature_index: int,
        feature_value: int,
        computed_metric: np.float32
    ):
        self.left_node = None
        self.right_node = None
        self.feature_index = feature_index
        self.feature_value = feature_value
        self.computed_metric = computed_metric

    def _determine_decision (self, hyperparameters: dict[str, Union[str, np.int32, np.float32]]):
        pass

class DecisionNode (Node):
    pass

class LeafNode (Node):
    pass

class DecisionTreeClassifier (DecisionNode, LeafNode, BaseEstimator, ClassifierMixin):
    __parameter_constraints__ = {
        "split_metric": (str),
        "split_type": (str),
        "max_depth": (np.int32),
        "min_samples_leaf": (np.int32),
        "min_information_gain": (np.float32),
        "max_leaf_nodes": (np.int32)
    }

    def __init__ (
        self,
        split_metric: str = "gini",
        split_type: str = None,
        max_depth: np.int32 = 10,
        min_samples_leaf: np.int32 = 30,
        min_information_gain: np.float32 = 1e-4,
        max_leaf_nodes: np.int32 = 10
    ):
        self.split_metric = split_metric
        self.split_type = split_type
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.max_leaf_nodes = max_leaf_nodes

    def _compute_class_probability (self, X: np.ndarray) -> np.ndarray:
        """
        The method computed the probability for each unique class

        Parameters:
            X (np.ndarray): The entire training dataset including the ground truths

        Returns:
            unique_class_probabilities (np.ndarray): Class probabilities for unique classes
        """
        _, label_count = np.unique(X[:, -1], return_counts=True)
        total_lable_count = len(X[:, -1])
        return [label_count[index] / total_lable_count for index in range(len(label_count))]
    
    def _compute_impurity (self, X: np.ndarray) -> np.float32:
        unique_probabilities = self._compute_class_probability(X[:, -1])
        computed_impurity = 1 - np.sum(np.square(unique_probabilities))
        return computed_impurity
    
    def _compute_entropy (self, X: np.ndarray) -> np.float32:
        unique_probabilities = self._compute_class_probability(X[:, -1])
        computed_entropy = -(np.sum(unique_probabilities * np.log2(unique_probabilities)))
        return computed_entropy
    
    def _compute_log_loss (self, Y_true: np.ndarray, Y_probabilities: np.ndarray) -> np.float32:
        log_loss_eq = (Y_true * np.log(Y_probabilities)) + (1 - Y_true) * np.log(1 - Y_probabilities)
        log_loss = 1 / len(Y_probabilities) * np.sum(log_loss_eq)
        return log_loss
    
    def _evaluate_split_type (self, X: np.ndarray):
        if self.split_metric == "gini":
            return self._compute_impurity(X)
        elif self.split_metric == "entropy":
            return self._compute_entropy(X)
        else:
            return self._compute_log_loss() # Volatile code
        # TODO: Correct the data that's being passed in the computing gini and entropy    
    
    def _determine_split_type (self, X: np.ndarray) -> np.ndarray:
        feature_indices_range = list(range(X.shape[1] - 1))

        if self.split_metric == "sqrt":
            feature_indices = np.random.choice(
                feature_indices_range, 
                size=int(np.sqrt(len(feature_indices_range)))
            )
        elif self.split_metric == "log2":
            feature_indices = np.random.choice(
                feature_indices_range, 
                size=int(np.log2(len(feature_indices_range)))
            )
        else:
            feature_indices = feature_indices_range

        return feature_indices
        
    def _generate_thresholds (self, X: np.ndarray):
        feature_indices = self._determine_split_type(X)
        for feature_index in feature_indices:
            yield feature_index, np.percentile(X[:, feature_index], q=np.arange(25, 100, 25))

    def _split_data (self, X: np.ndarray):
        for feat_index, percentile_threshold in self._generate_thresholds(X):
            group_below_threshold = np.where(X[:, feat_index] < percentile_threshold)
            group_above_threshold = np.where(X[:, feat_index] > percentile_threshold)

            yield feat_index, percentile_threshold, group_below_threshold, group_above_threshold
        
    def _create_node (self, X: np.ndarray):
        current_looped_index = None
        current_percentile_index = None

        minimum_computed_below_threshold = None
        minimum_computed_below_threshold = None
        minimum_temporary_above_threshold = None
        minimum_temporary_above_threshold = None

        # Compute the below_threshold group
        # TODO: Create a if elif else ladder checking for data information theory checking
        # TODO: If current proves to be greater than existing minumum, replace

        for feat_index, percent_thresh, below_group_threshold, above_group_threshold in self._split_data(X):
            current_looped_index = feat_index
            current_percentile_index = percent_thresh

            # Comparing the below_threshold_group
            if self._evaluate_split_type(below_group_threshold) < minimum_computed_below_threshold:
                minimum_computed_below_threshold = below_group_threshold
                self.left_node = Node(
                    current_looped_index,
                    current_percentile_index,
                    self._evaluate_split_type(below_group_threshold)
                )
            else:
                continue

            # Comparing the above_threshold_group
            if self._evaluate_split_type(above_group_threshold) < minimum_temporary_above_threshold:
                minimum_temporary_above_threshold = above_group_threshold
                self.right_node = Node(
                    current_looped_index,
                    current_percentile_index,
                    self._evaluate_split_type(above_group_threshold)
                )
            else:
                continue
