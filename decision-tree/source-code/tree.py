from typing import Any, Union
import numpy as np
import pandas as pd
from base.EstimatorClass import BaseEstimator, ClassifierMixin

class DecisionNode:
    def __init__ (
        self,
        feature_index: int = None,
        numeric_feature_condition: Union[int, float] = None,
        categorical_feature_condition: int = None,
        min_computed_metric_below = None,
        min_computed_metric_above = None,
        min_iterated_below_group = None,
        min_iterated_above_group = None
    ):
        self.feature_index = feature_index
        self.numeric_feature_condition = numeric_feature_condition
        self.categorical_feature_condition = categorical_feature_condition
        self.min_computed_metric_below = min_computed_metric_below
        self.min_computed_metric_above = min_computed_metric_above
        self.min_iterated_below = min_iterated_below_group
        self.min_iterated_above = min_iterated_above_group
        self.left_node = None
        self.right_node = None

class LeafNode:
    def __init__ (self, computed_probabilities: np.ndarray):
        self.tree_computed_probabilities = computed_probabilities

    def compute_argmax (self):
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
            for feature_percentile_range in np.percentile(X[:, feature_index], q=np.arange(25, 100, 25)):
                yield feature_index, feature_percentile_range

    def _split_data (self, X: np.ndarray):
        for feat_index, percentile_threshold in self._generate_thresholds(X):
            group_below_threshold = X[:, feat_index][X[:, feat_index] < percentile_threshold]
            group_above_threshold = X[:, feat_index][X[:, feat_index] > percentile_threshold]

            yield feat_index, percentile_threshold, group_below_threshold, group_above_threshold
        
    def _create_node (self, X: np.ndarray):
        below_feature_index, above_feature_index = 0
        
        min_computed_metric_below, min_computed_metric_above = 0
        min_below_threshold, min_above_threshold = None
        min_iterated_below_group, min_iterated_above_group = None

        # TODO: Add condition to check if feature is continuous or categorical

        for feat_index, percentile_thresh, below_group, above_group in self._split_data(X):
            temp_metric_below = self._evaluate_split_type(below_group[:, -1])
            temp_metric_above = self._evaluate_split_type(above_group[:, -1])

            if temp_metric_below < min_computed_metric_below:
                below_feature_index = feat_index
                min_below_threshold = percentile_thresh
                min_computed_metric_below = temp_metric_below
                min_iterated_below_group = below_group

            if temp_metric_below < min_computed_metric_below:
                above_feature_index = feat_index
                min_above_threshold = feat_index
                min_computed_metric_above = temp_metric_above
                min_iterated_above_group = above_group

        self.left_node = DecisionNode(
            below_feature_index,
            min_below_threshold,
            min_computed_metric_below=min_computed_metric_below,
            min_iterated_below_group=min_iterated_below_group
        )

        self.right_node = DecisionNode(
            above_feature_index,
            min_above_threshold,
            min_computed_metric_above=min_computed_metric_above,
            min_iterated_above_group=min_iterated_above_group
        )
