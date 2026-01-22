from typing import Any, Union
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from validator.validator import DatasetValidation
from validator import ParameterValidator
from base.BaseEstimator import BaseEstimator
from base.ClassifierMixin import ClassifierMixin

class DecisionNode:
    def __init__ (
        self,
        split_index: int,
        split_feat_num_condition: int,
        split_feat_cat_condition: Union[int, str],
        information_gain: float
    ):
        self._node_is_decision = True
        self._split_index = split_index
        self._feat_num_condition = split_feat_num_condition
        self._feat_cat_condition = split_feat_cat_condition
        self._information_gain = information_gain
        self._left_node = None
        self._right_node = None

class LeafNode:
    def __init__ (self, computed_probabilities: np.ndarray):
        self._node_is_leaf = True
        self._leaf_node_amount = 0
        self.tree_computed_probabilities = computed_probabilities

    def compute_argmax (self):
        return np.argmax(self.tree_computed_probabilities)

class DecisionTreeClassifier (BaseEstimator, ClassifierMixin):
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
        Y = X[:, -1]
        label, label_count = np.unique(Y, return_counts=True)
        return np.asarray([label_count / len(Y)])
    
    def _compute_impurity (self, X: np.ndarray) -> np.float32:
        """
        Computes the gini impurity using the formula: Gini = 1 - Σ(pᵢ²)

        Parameters:
            X (np.ndarray): The entire or subset of the dataset to compute the impurity on

        Returns:
            computed_impurity (np.float32): The floating point computed gini impurity
        """
        unique_probabilities = self._compute_class_probability(X)
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
        unique_probabilities = self._compute_class_probability(X)
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
        main_data_impurity = self._determine_impurity_metric(X)
        left_subset_impurity = self._determine_impurity_metric(left_subset)
        right_subset_impurity = self._determine_impurity_metric(right_subset)

        left_subset_weighted = len(left_subset) / len(X) * left_subset_impurity
        right_subset_weighted = len(right_subset) / len(X) * right_subset_impurity
        
        return main_data_impurity - (left_subset_weighted + right_subset_weighted)
    
    def _determine_impurity_metric (self, X: np.ndarray):
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
    
    def _determine_feature_split_metric (self, X: np.ndarray) -> np.ndarray:
        """
        Splits N amount of features depending on what split metric will be used

        Parameters:
            X (np.ndarray): The entire dataset to be splitted by N features

        Returns:
            feature_indices (np.ndarray): The list of feature indices to be used when generating thresholds
        """
        feature_indices_range = list(range(X.shape[1] - 1))

        if self.split_type == "sqrt":
            feature_indices = np.random.choice(
                feature_indices_range, 
                size=int(np.sqrt(len(feature_indices_range)))
            )
        elif self.split_type == "log2":
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
                unique_values = np.unique(X[:, num_feat])
                computed_midpoint_thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                for threshold in computed_midpoint_thresholds:
                    yield (
                        "numerical",
                        num_feat,
                        threshold,
                        X[X[:, num_feat] > threshold],
                        X[X[:, num_feat] < threshold]
                    )
        if categorical_features:
            for cat_feat in categorical_features:
                for unique_feat in np.nditer(np.unique(X[:, cat_feat])):
                    yield (
                        "categorical",
                        cat_feat,
                        unique_feat,
                        X[X[:, cat_feat] == unique_feat],
                        X[X[:, cat_feat] != unique_feat]
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

        if self.split_metric and self.categorical_features:
            numeric_features = self._determine_split_type(numeric_dataset)
            categorical_features = self._determine_split_type(categorical_dataset)
        else:
            numeric_features = self._determine_split_type(X)

        if self.categorical_features:
            yield from self._split_yield(X, numeric_features, categorical_features)

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
            if split_num_condition:
                instantiated_decision_node = DecisionNode(
                    split_index=split_index,
                    split_feat_num_condition=split_num_condition,
                    split_feat_cat_condition=None,
                    information_gain=information_gain
                )
            if split_cat_condition:
                instantiated_decision_node = DecisionNode(
                    split_index=split_index,
                    split_feat_num_condition=None,
                    split_feat_cat_condition=split_cat_condition,
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
        best_index = None
        best_num_split_condition = None
        best_unique_split_condition = None
        best_computed_left_split: np.ndarray = None
        best_computed_right_split: np.ndarray = None

        # Tree depth & min_samples_split hyperparameter check
        if recursive_tree_depth == self.max_depth or len(X) <= self.min_samples_split:
            leaf_node = self._create_node(
                computed_class_probabilities=self._compute_class_probability(X), 
                create_leaf_node=True
            )
            return leaf_node

        for feat_type, feat_index, condition, group_above_condition, group_below_condition in self._split_data(X):
            if feat_type == "numerical":
                num_feat_information_gain = self._compute_information_gain(X, group_above_condition, group_below_condition)
                if num_feat_information_gain > best_computed_information_gain:
                    best_index = feat_index
                    best_computed_information_gain = num_feat_information_gain
                    best_num_split_condition = condition
                    best_computed_left_split = group_above_condition
                    best_computed_right_split = group_below_condition
                    best_cat_split_condition = None
                else:
                    continue

            if feat_type == "categorical":
                cat_feat_information_gain = self._compute_information_gain(X, group_above_condition, group_below_condition)
                if cat_feat_information_gain > best_computed_information_gain:
                    best_index = feat_index
                    best_computed_information_gain = cat_feat_information_gain
                    best_unique_split_condition = condition
                    best_computed_left_split = group_above_condition
                    best_computed_right_split = group_below_condition
                    best_unique_split_condition = None
                else:
                    continue

        # min_information_gain hyperparameter check
        if best_computed_information_gain < self.min_information_gain:
            instantiated_leaf_node = self._create_node(
                computed_class_probabilities=self._compute_class_probability(X), 
                create_leaf_node=True
            )
            return instantiated_leaf_node

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
            split_index=best_index,
            split_num_condition=best_num_split_condition,
            split_cat_condition=best_unique_split_condition,
            information_gain=best_computed_information_gain,
            create_decision_node=True
        )

        instantiated_decision_node._left_node = self._build_decision_tree(best_computed_left_split, recursive_tree_depth + 1)
        instantiated_decision_node._right_node = self._build_decision_tree(best_computed_right_split, recursive_tree_depth + 1)

        return instantiated_decision_node

    def _inference_traversal_tree (self, X: np.ndarray, root_node: Union[DecisionNode, LeafNode]):
        """
        Inferences a single data point by traversing the tree until it reaches a leaf node

        Parameters:
            X (np.ndarray): A single datapoint to be inferenced

        Returns:
            (list[np.float32]): Computed class probabilities inside leaf node
        """
        while root_node._node_is_decision:
            if root_node._feat_num_condition and X[root_node._split_index] > root_node._feat_num_condition:
                self._inference_traversal_tree(X, root_node._left_node)
            else:
                self._inference_traversal_tree(X, root_node._right_node)

            if root_node._feat_cat_condition and X[root_node._split_index] == root_node._feat_cat_condition:
                self._inference_traversal_tree(X, root_node._left_node)
            else:
                self._inference_traversal_tree(X, root_node._right_node)

            if root_node._node_is_leaf:
                return True, root_node.compute_argmax()

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
        for row in X:
            leaf_node, predicted_class = self._inference_traversal_tree(row, self._root_node)
            if leaf_node:
                inferenced_elements_array.append(predicted_class)
            
        return np.asarray(inferenced_elements_array)

def main ():
    # --- Load data generated by python-utilities/generator-files/generator.py ---
    # We rely on the already-generated CSVs under python-utilities/test-data/classification-data.
    # If you want to (re)generate them, run generator.py separately.

    repo_root = Path(__file__).resolve().parents[2]
    util_root = repo_root / "python-utilities"
    classification_dir = util_root / "test-data" / "classification-data"

    train_x_path = classification_dir / "train_x.csv"
    train_y_path = classification_dir / "train_y.csv"
    test_x_path = classification_dir / "test_x.csv"
    test_y_path = classification_dir / "test_y.csv"

    if not (train_x_path.exists() and train_y_path.exists() and test_x_path.exists() and test_y_path.exists()):
        raise FileNotFoundError(
            "Missing classification CSVs. Generate them with: "
            "python python-utilities/generator-files/generator.py --dataset-type classification"
        )

    train_x_df = pd.read_csv(train_x_path)
    train_y_df = pd.read_csv(train_y_path)
    test_x_df = pd.read_csv(test_x_path)
    test_y_df = pd.read_csv(test_y_path)

    X_full = pd.concat([train_x_df, test_x_df], axis=0, ignore_index=True).to_numpy(dtype=np.float32)
    y_full = pd.concat([train_y_df, test_y_df], axis=0, ignore_index=True).to_numpy().reshape(-1)

    # --- Split into train/test (done here as requested) ---
    rng = np.random.default_rng(42)
    indices = rng.permutation(X_full.shape[0])
    split_idx = int(0.8 * X_full.shape[0])
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X_full[train_idx]
    y_train = y_full[train_idx]
    X_test = X_full[test_idx]
    y_test = y_full[test_idx]

    # Custom tree expects labels in last column during fit
    train_combined = np.column_stack([X_train, y_train])

    # --- Train + evaluate custom model ---
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score
    except Exception as e:
        raise ImportError("scikit-learn is required for metrics (accuracy/precision/recall)") from e

    print("\n[Custom DecisionTreeClassifier]")
    try:
        tree_instance = DecisionTreeClassifier(random_state=42)
        tree_instance.fit(train_combined)
        y_pred = tree_instance.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
    except Exception as e:
        # IMPORTANT: Do not fall back to sklearn as the default model.
        print(f"Custom DecisionTreeClassifier failed: {type(e).__name__}: {e}")

    # --- Optional comparison vs sklearn's DecisionTreeClassifier (not a fallback) ---
    print("\n[Sklearn DecisionTreeClassifier comparison]")
    try:
        from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier

        sk_model = SkDecisionTreeClassifier(random_state=42)
        sk_model.fit(X_train, y_train)
        sk_pred = sk_model.predict(X_test)

        sk_acc = accuracy_score(y_test, sk_pred)
        sk_prec = precision_score(y_test, sk_pred, average="macro", zero_division=0)
        sk_rec = recall_score(y_test, sk_pred, average="macro", zero_division=0)
        print(f"Accuracy : {sk_acc:.4f}")
        print(f"Precision: {sk_prec:.4f}")
        print(f"Recall   : {sk_rec:.4f}")
    except Exception as e:
        print(f"Skipped sklearn comparison: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
