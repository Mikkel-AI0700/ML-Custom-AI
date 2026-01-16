from typing import Any, Union
import numpy as np
import pandas as pd
from validator.validator import DatasetValidation
from validator import ParameterValidator
from base.BaseEstimator import BaseEstimator
from base.ClassifierMixin import ClassifierMixin

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

class LeafNode:
    def __init__ (self, computed_probabilities: np.ndarray):
        self._node_is_leaf = False
        self._leaf_node_amount = 0
        self.tree_computed_probabilities = computed_probabilities

    def compute_argmax (self):
        return np.argmax(self.tree_computed_probabilities)

class DecisionTreeClassifier (DecisionNode, LeafNode, BaseEstimator, ClassifierMixin):
    """
    DecisionTreeClassifier class uses the concept of a greedy algorithm to search
    from top to bottom what best separates all the unique classes in the dataset

    Parameters:
        split_metric (str): The splitting criteria using Information Theory when computing the impurity or randomness of the classes
        split_type (str): Controls how much features to be used after splitting
        max_depth (int): The maximum allowable depth of recursively created subtrees
        max_leaf_nodes (int): The maximum allowable amount of created leaf nodes in the tree
        min_samples_leaf (int): The minimum allowable amount of samples needed inside a node to convert to leaf
        min_samples_split (int): The minimum allowable amount of samples needed inside a node to split
        min_information_gain (float): The minimum allowable information gain per iteration. If below, will convert to leaf node.
        random_state (int): Controls computer's randomness. Ensures reproducibility

    Returns:
        DecisionTreeClassifier (object): The instantiated DecisionTreeClassifier
    """
    __parameter_constraints__ = {
        "split_metric": (str),
        "split_type": (str),
        "min_samples_leaf": (np.int32),
        "min_samples_split": (np.int32),
        "max_leaf_nodes": (np.int32),
        "min_information_gain": (np.float32)
    }

    def __init__ (
        self,
        split_metric: str = "gini",
        split_type: str = None,
        max_depth: np.int32 = 10,
        max_leaf_nodes: np.int32 = 10,
        min_samples_leaf: np.int32 = 30,
        min_samples_split: np.int32 = 10,
        min_information_gain: np.float32 = 1e-4,
        categorical_features: Union[list[int] | list[str]] = None,
        random_state: int = 42
    ):
        self.split_metric = split_metric
        self.split_type = split_type
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.categorical_features = categorical_features
        self.random_state = random_state
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
        return np.asarray([label_count[index] / total_lable_count for index in range(len(label_count))])
    
    def _compute_impurity (self, X: np.ndarray) -> np.float32:
        """
        Computes the gini impurity using the formula: Gini = 1 - Σ(pᵢ²)

        Parameters:
            X (np.ndarray): The entire or subset of the dataset to compute the impurity on

        Returns:
            computed_impurity (np.float32): The floating point computed gini impurity
        """
        unique_probabilities = self._compute_class_probability(X[:, -1])
        computed_impurity = 1 - np.sum(np.square(unique_probabilities))
        return computed_impurity
    
    def _compute_entropy (self, X: np.ndarray) -> np.float32:
        """
        Computed the entropy randomness using the formula: H(X) = -Σ p(xᵢ) log₂(p(xᵢ))

        Parameters:
            X (np.ndarray): THe entire or subset of the dataset to compute the impurity on

        Returns:
            computed_entropy (np.float32): The floating point computed gini impurity
        """
        unique_probabilities = self._compute_class_probability(X[:, -1])
        computed_entropy = -(np.sum(unique_probabilities * np.log2(unique_probabilities)))
        return computed_entropy
    
    def _compute_log_loss (self, Y_true: np.ndarray, Y_probabilities: np.ndarray) -> np.float32:
        """
        Computed the log loss

        Parameters:
            Y_true (np.ndarray): The ground truths from the dataset
            Y_probabilities (np.ndarray): The computed probabilities of the specific class

        Returns:
            log_loss (np.float32): The floating point computed log loss

        Warning:
            _comnpute_log_loss still remains unstable and not yet supported.
        """
        log_loss_eq = (Y_true * np.log(Y_probabilities)) + (1 - Y_true) * np.log(1 - Y_probabilities)
        log_loss = 1 / len(Y_probabilities) * np.sum(log_loss_eq)
        return log_loss
    
    def _compute_information_gain (self, X: np.ndarray, left_subset: np.ndarray, right_subset: np.ndarray):
        """
        Computes the information gain using the formula: Gain(S, A) = Entropy(S) - Σ ( |Sv| / |S| ) * Entropy(Sv)

        Parameters:
            X (np.ndarray): The entire dataset to be used for computing the information gain
            left_subset (np.ndarray): The best subset of data with the best information gain in the left node
            right_subset (np.ndarray): The best subset of data with the best information gain in the right node

        Returns:
            information_gain (np.float32): The floating point computed information gain
        """
        main_data_impurity = self._evaluate_split_type(X)
        left_subset_impurity = self._evaluate_split_type(left_subset)
        right_subset_impurity = self._evaluate_split_type(right_subset)

        left_subset_weighted = len(left_subset) / len(X) * left_subset_impurity
        right_subset_weighted = len(right_subset) / len(X) * right_subset_impurity
        
        return main_data_impurity - (left_subset_weighted + right_subset_weighted)
    
    def _evaluate_split_type (self, X: np.ndarray):
        """
        Will evaluate the type of splitting metric type on the entire or just a subset of the dataset

        Parameters:
            X (np.ndarray): The entire of the just the subset of the dataset to be used for computing

        Returns:
            metric_score (np.float32): Coming from _compute_impurity and _compute_entropy, will return np.float32 score
        """
        if self.split_metric == "gini":
            return self._compute_impurity(X)
        elif self.split_metric == "entropy":
            return self._compute_entropy(X)
        else:
            return self._compute_log_loss() # WARNING: Very volatile code. Don't be stupid and run this line ;)
    
    def _determine_split_type (self, X: np.ndarray) -> np.ndarray:
        """
        Splits N amount of features depending on what split metric will be used

        Parameters:
            X (np.ndarray): The entire dataset to be splitted by N features

        Returns:
            feature_indices (np.ndarray): The list of feature indices to be used when generating thresholds
        """
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
        
    def _split_yield (self, X: np.ndarray, numeric_features: list[int], categorical_features: list[int]):
        if numeric_features:
            for num_feat in numeric_features:
                for num_feat_index, num_feat_value in enumerate(np.nditer(X[:, num_feat])):
                    midpoint_value = (num_feat_value[num_feat_index] + num_feat_value[num_feat_index + 1]) / 2
                    yield (
                        "numerical",
                        midpoint_value,
                        np.where(X[:, num_feat] > midpoint_value),
                        np.where(X[:, num_feat] < midpoint_value)
                    )

        if categorical_features:
            for cat_feat in categorical_features:
                for unique_feat in np.nditer(np.unique(X[:, cat_feat])):
                    yield (
                        "categorical",
                        unique_feat,
                        np.where(X[:, cat_feat] == unique_feat),
                        np.where(X[:, cat_feat] != unique_feat)
                    )

    def _split_data (self, X: np.ndarray):
        """
        Will iterate using the yielded values from _generate_thresholds, then split it
        into two groups: below the threshold and above the threshold

        Parameters:
            X (np.ndarray): The entire or just the subset of the dataset to be splitted

        Yields:
            feat_index (int): The current iterated feature index
            percentile_threshold (list[ints]): The current percentile in the iteration
            filtered_below_threshold (np.ndarray): The ndarray containing all the values that fall under the percentile threshold
            filtered_above_threhsold (np.ndarray): The ndarray containing all the values that is above the percentile threhsold
        """
        if self.categorical_features:
            numeric_dataset = X[:, ~self.categorical_features]
            categorical_dataset = X[:, self.categorical_features]

        if self.split_metric:
            numeric_features = self._determine_split_type(numeric_dataset)
            categorical_features = self._determine_split_type(categorical_dataset)

        if self.categorical_features:
            num_feat_index, num_feat_percentile = self._split_yield(X, numeric_features)
            
        else:
            pass

    def _create_node (
        self, 
        split_index: int,
        split_num_condition: Union[int, float],
        split_cat_condition: Union[int, float],
        information_gain: float,
        computed_class_probabilities: np.ndarray,
        create_decision_node: bool = False,
        create_leaf_node: bool = False,
    ) -> Union[DecisionNode, LeafNode]:
        """
        Will either create and return a DecisionNode or LeafNode, depending on the 
        status of the recursive tree builder function

        Parameters:
            split_index (int): Split index unique to the iteration
            split_num_condition (Union[int, float]): The numerical condition that splitted the data unique to the iteration
            split_cat_condition (Union[int, float]): The categorical conditoon that splitted the data unique to that iteration
            information_gain (float): The best computed information gain unique to that iteration
            computed_class_probabilities (np.ndarray): The computed class probabilities when node reaches a certain level
            create_decision_node (bool): Boolean flag to check that signals a DecisionNode creation
            create_leaf_node (bool): Boolean flag to check that signals a LeafNode creation

        Returns:
            instantiated_node (Union[DecisionNode, LeafNode]): The instantiated object of either the DecisionNode or LeafNode
        """
        if create_decision_node:
            instantiated_decision_node = DecisionNode(
                split_feat_index=split_index,
                split_feat_num_condition=split_num_condition,
                split_feat_cat_condition=None,
                information_gain=information_gain
            )
            return instantiated_decision_node

        if create_leaf_node:
            instantiated_leaf_node = LeafNode(computed_class_probabilities)
            instantiated_leaf_node._leaf_node_amount = instantiated_leaf_node._leaf_node_amount + 1
            instantiated_leaf_node._node_is_leaf = True
            return instantiated_leaf_node

    def _build_decision_tree (self, X: np.ndarray, recursive_tree_depth: int = 1):
        """
        The main recursive tree builder function that recursively builds the main decision tree
        by building the left and right subtrees

        Parameters:
            X (np.ndarray): The entire or just the subset of the dataset that will be used when training
            recursive_tree_depth (int): Internal tracking variable. Used to track the depth of the corresponding subtree

        Returns:
            (Union[DecisionNode, LeafNode]): Returns DecisionNode if no stopping criteria is met, else returns LeafNode
        """
        best_computed_information_gain = 0
        best_split_index = 0
        best_split_condition = 0
        best_computed_left_split: np.ndarray = None
        best_computed_right_split: np.ndarray = None

        # Tree depth & min_samples_split hyperparameter check
        if recursive_tree_depth == self.max_depth or len(X) <= self.min_samples_split:
            leaf_node = self._create_node(
                computed_class_probabilities=self._compute_class_probability(X), 
                create_leaf_node=True
            )
            return leaf_node

        for feat_index, percentile_threshold, below_group, above_group in self._split_data(X):
            # TODO: Add logic that will check if the decision tree is running for the first time
            computed_information_gain = self._compute_information_gain(X, below_group, above_group)
            if computed_information_gain > best_computed_information_gain:
                best_split_index = feat_index
                best_split_condition = percentile_threshold
                best_computed_left_split = below_group
                best_computed_right_split = above_group

        # min_information_gain hyperparameter check
        if best_computed_information_gain < self.min_information_gain:
            instantiated_leaf_node = self._create_node(
                computed_class_probabilities=self._compute_class_probability(X), 
                create_leaf_node=True
            )
            return instantiated_decision_node

        # min_samples_leaf hyperparameter check
        if (best_computed_left_split.shape[0] < self.min_samples_leaf or
            best_computed_right_split.shape[0] < self.min_samples_leaf
        ):
            instantiated_leaf_node = self._create_node(
                computed_class_probabilities=self._compute_class_probability(X), 
                create_leaf_node=True
            )
            return instantiated_leaf_node

        instantiated_decision_node = self._create_node(
            split_index=best_split_index,
            split_num_condition=best_split_condition,
            split_cat_condition=None,
            information_gain=best_computed_information_gain,
            create_decision_node=True
        )

        instantiated_decision_node.left_node = self._build_decision_tree(best_computed_left_split, recursive_tree_depth + 1)
        instantiated_decision_node.right_node = self._build_decision_tree(best_computed_right_split, recursive_tree_depth + 1)

        return instantiated_decision_node

    def _inference_traversal_tree (self, X: np.ndarray, root_node: Union[DecisionNode, LeafNode]):
        """
        Inferences a single data point by traversing the tree until it reaches a leaf node

        Parameters:
            X (np.ndarray): A single datapoint to be inferenced

        Returns:
            (list[np.float32]): Computed class probabilities inside leaf node
        """
        if root_node._node_is_leaf:
            return True, root_node.compute_argmax()

        if root_node.feat_num_condition < X:
            return self._inference_traversal_tree(X, root_node.left_node)
        else:
            return self._inference_traversal_tree(X, root_node.right_node)

    def fit (self, X: np.ndarray):
        """
        Train the entire DecisionTreeClassifier using the entire dataset

        Parameters:
            X (np.ndarray): The entire dataset to be used for training

        Returns:
            None
        """
        self._dset_validator.validate_existence(X)
        self._dset_validator.validate_types(X)
        self._root_node = self._build_decision_tree(X)

    def predict (self, X: np.ndarray):
        """
        Loops over the data samples then returns the class probability
        that corresponds to the predicted class

        Parameters:
            X (np.ndarray): The entire dataset to be used for inference

        Returns:
            inferenced_elements_array (np.ndarray): Contains the predicted elements using the stored class probabilities
        """
        inferenced_elements_array = []
        for data in np.nditer(X):
            leaf_node, prediction_probability_array = self._inference_traversal_tree(data, self._root_node)
            if leaf_node:
                inferenced_elements_array.append(np.argmax(prediction_probability_array))
            
        return np.asarray(inferenced_elements_array)
        # TODO: Fix the logic where it will just get the highest argmax value of the stored
        # TODO: class probabilities

def main ():
    tree_instance = DecisionTreeClassifier()
