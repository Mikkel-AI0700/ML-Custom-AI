from typing import Any, Union
import numpy as np
import pandas as pd
from validator.validator import DatasetValidation
from validator import ParameterValidator
from base.EstimatorClass import BaseEstimator, ClassifierMixin

class DecisionNode:
    def __init__ (
        self,
        split_feat_index: int,
        split_feat_num_condition: int,
        split_feat_cat_condition: Union[int, str],
        information_gain: float
    ):
        self.node_is_decision = False
        self.feat_index = split_feat_index
        self.feat_num_condition = split_feat_num_condition
        self.feat_cat_condition = split_feat_cat_condition
        self.information_gain = information_gain
        self.left_node = None
        self.right_node = None
        self._left_node_depth = 0
        self._right_node_depth = 0

class LeafNode:
    def __init__ (self, computed_probabilities: np.ndarray):
        self.node_is_leaf = False
        self.tree_computed_probabilities = computed_probabilities

    def compute_argmax (self):
        return np.argmax(self.tree_computed_probabilities)

class DecisionTreeClassifier (DecisionNode, LeafNode, BaseEstimator, ClassifierMixin):
    __parameter_constraints__ = {
        "split_metric": (str),
        "split_type": (str),
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
        max_leaf_nodes: np.int32 = 10,
    ):
        self.split_metric = split_metric
        self.split_type = split_type
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.max_leaf_nodes = max_leaf_nodes
        self._root_node: Union[DecisionNode, LeafNode] = None
        self._recursive_max_depth = 0
        self._recursive_max_leaf_nodes = 0
        self._recursive_min_information_gain = 0.0
        self._dset_validator = DatasetValidation()
        self._param_validator = ParameterValidator()

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
    
    def _compute_information_gain (self, X: np.ndarray, left_subset: np.ndarray, right_subset: np.ndarray):
        main_data_impurity = self._evaluate_split_type(X)
        left_subset_impurity = self._evaluate_split_type(left_subset)
        right_subset_impurity = self._evaluate_split_type(right_subset)

        left_subset_weighted = len(left_subset) / len(X) * left_subset_impurity
        right_subset_weighted = len(right_subset) / len(X) * right_subset_impurity
        
        return main_data_impurity - (left_subset_weighted + right_subset_weighted)
    
    def _evaluate_split_type (self, X: np.ndarray):
        if self.split_metric == "gini":
            return self._compute_impurity(X)
        elif self.split_metric == "entropy":
            return self._compute_entropy(X)
        else:
            return self._compute_log_loss() # WARNING: Very volatile code. Don't be stupid and run this line ;)
    
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
            group_below_threshold = X[:, feat_index] < percentile_threshold
            group_above_threshold = X[:, feat_index] > percentile_threshold

            filtered_below_threshold = X[group_below_threshold]
            filtered_above_threshold = X[group_above_threshold]

            yield feat_index, percentile_threshold, filtered_below_threshold, filtered_above_threshold
        
    def _inference_traversal_tree (self, X: np.ndarray, root_node: Union[DecisionNode, LeafNode]):
        if root_node.node_is_leaf:
            return True, root_node.compute_argmax()

        if root_node.feat_num_condition < X:
            return self._inference_traversal_tree(X, root_node.left_node)
        else:
            return self._inference_traversal_tree(X, root_node.right_node)

    def _build_decision_tree (self, X: np.ndarray, computing_left = False, computing_right = False):
        best_computed_information_gain = 0
        best_split_index = 0
        best_split_condition = 0
        best_computed_left_split = None
        best_computed_right_split = None

        for feat_index, percentile_threshold, below_group, above_group in self._split_data(X):
            computed_information_gain = self._compute_information_gain(X, below_group, above_group)
            if computed_information_gain > best_computed_information_gain:
                best_split_index = feat_index
                best_split_condition = percentile_threshold
                best_computed_left_split = below_group
                best_computed_right_split = above_group

        temp_decision_node = DecisionNode(
            best_split_index,
            best_split_condition,
            None,
        )

        if (self._recursive_max_depth == self.max_depth or
            self._recursive_max_leaf_nodes == self.max_leaf_nodes or
            self._recursive_min_information_gain > self.min_information_gain
        ):
            if computing_left:
                self._left_node_depth = self._left_node_depth + 1
                return LeafNode(self._compute_class_probability(np.asarray(best_computed_left_split)))
            if computing_right:
                self._right_node_depth = self._right_node_depth + 1
                return LeafNode(self._compute_class_probability(np.asarray(best_computed_right_split)))
        
        temp_decision_node.left_node = self._build_decision_tree(best_computed_left_split, computing_left=True)
        temp_decision_node.right_node = self._build_decision_tree(best_computed_right_split, computing_right=True)

        return temp_decision_node
    
    def fit (self, X: np.ndarray):
        self._dset_validator.validate_existence(X)
        self._dset_validator.validate_types(X)
        self._root_node = self._build_decision_tree(X)

    def predict (self, X: np.ndarray):
        inferenced_elements_array = []
        for data in np.nditer(X):
            leaf_node, prediction_probability_array = self._inference_traversal_tree(data, self._root_node)
            if leaf_node:
                inferenced_elements_array.append(np.argmax(prediction_probability_array))
            
        return np.asarray(inferenced_elements_array)
        # TODO: Fix the logic where it will just get the highest argmax value of the stored
        # TODO: class probabilities

def main ():
    pass
