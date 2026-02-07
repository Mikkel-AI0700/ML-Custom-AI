from typing import Any, Union
import numpy as np
import pandas as pd
from validator.DatasetValidation import DatasetValidation
from validator import ParameterValidator
from base.BaseEstimator import BaseEstimator
from base.ClassifierMixin import ClassifierMixin

class DecisionNode:
    """Internal decision node used by :class:`DecisionTreeClassifier`.

    A decision node stores a single split rule and pointers to the left/right
    subtrees. The rule can be either numerical (threshold) or categorical
    (equality check), depending on which condition is populated.

    Parameters
    ----------
    split_index : int
        Feature index used for the split.
    split_feat_num_condition : int or float or None
        Numerical threshold used for the split. Samples with
        ``X[split_index] > split_feat_num_condition`` follow the left subtree in
        this implementation.
    split_feat_cat_condition : int or str or None
        Categorical value used for the split. Samples with
        ``X[split_index] == split_feat_cat_condition`` follow the left subtree in
        this implementation.
    information_gain : float
        Information gain associated with the chosen split.

    Attributes
    ----------
    _left_node : DecisionNode or LeafNode or None
        Left subtree.
    _right_node : DecisionNode or LeafNode or None
        Right subtree.
    """
    def __init__ (
        self,
        split_index: int,
        split_feat_num_condition: int,
        split_feat_cat_condition: Union[int, str],
        information_gain: float
    ):
        self._node_is_decision = True
        self._node_is_leaf = False
        self._split_index = split_index
        self._feat_num_condition = split_feat_num_condition
        self._feat_cat_condition = split_feat_cat_condition
        self._information_gain = information_gain
        self._left_node = None
        self._right_node = None

class LeafNode:
    """Internal leaf node used by :class:`DecisionTreeClassifier`.

    The leaf stores the class distribution observed at training time for the
    subset of samples reaching the leaf.

    Parameters
    ----------
    computed_probabilities : numpy.ndarray
        Class probability vector computed from the labels in the node's subset.

    Attributes
    ----------
    tree_computed_probabilities : numpy.ndarray
        Stored probability vector for the leaf.
    """
    def __init__ (self, computed_probabilities: np.ndarray):
        self._node_is_decision = False
        self._node_is_leaf = True
        self.tree_computed_probabilities = computed_probabilities

    def compute_argmax (self):
        """Return the predicted class index at this leaf.

        Returns
        -------
        int
            Index of the maximum probability class.
        """
        return np.argmax(self.tree_computed_probabilities)

class DecisionTreeClassifier (BaseEstimator, ClassifierMixin):
    """Greedy decision tree classifier.

    This is a custom implementation of a (binary) decision tree classifier that
    recursively selects a split maximizing information gain (according to the
    configured impurity metric) and grows subtrees until a stopping criterion is
    reached.

    The training API differs slightly from scikit-learn: during ``fit`` this
    implementation expects the target labels to be present in the last column of
    the provided training array.

    Parameters
    ----------
    split_metric : {'gini', 'entropy'}, default='gini'
        Impurity metric used to evaluate candidate splits.
    max_depth : numpy.int32, default=10
        Maximum recursion depth for the tree (root depth is 1 in this
        implementation).
    max_features : int or str or None, default=None
        Feature subsampling strategy.

        Notes
        -----
        The current implementation routes feature subsampling through
        ``_determine_feature_split_metric``. In that function, the values
        ``'sqrt'`` and ``'log2'`` are treated specially; otherwise, all features
        are considered.
    max_leaf_nodes : numpy.int32, default=10
        Maximum number of leaf nodes allowed.
        
        Notes
        -----
        This hyperparameter is currently stored but not enforced in the tree
        growth logic.
    min_samples_leaf : numpy.int32, default=30
        Minimum number of samples required in each child node after a split.
        If either side would contain fewer than ``min_samples_leaf`` samples,
        the split is rejected and the current node becomes a leaf.
    min_samples_split : numpy.int32, default=10
        Minimum number of samples required at a node to consider splitting.
    min_information_gain : numpy.float32, default=1e-4
        Minimum information gain required to accept a split. If the best gain is
        below this threshold, the current node becomes a leaf.
    numerical_features : list of int or list of str or None, default=None
        Indices (or names) of numerical features.

        Notes
        -----
        The splitter assumes numerical features are comparable and uses midpoint
        thresholds between unique values.
    categorical_features : list of int or list of str or None, default=None
        Indices (or names) of categorical features.
    random_state : int, default=42
        Random seed used for feature subsampling.

    Attributes
    ----------
    _root_node : DecisionNode or LeafNode or None
        Root of the trained tree.

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
        max_depth: np.int32 = 10,
        max_features: Union[int, str] = None,
        max_leaf_nodes: np.int32 = 10,
        min_samples_leaf: np.int32 = 30,
        min_samples_split: np.int32 = 10,
        min_information_gain: np.float32 = 1e-4,
        numerical_features: Union[list[int] | list[str]] = None,
        categorical_features: Union[list[int] | list[str]] = None,
        random_state: int = 42
    ):
        self.split_metric = split_metric
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.random_state = random_state
        self._root_node: Union[DecisionNode, LeafNode] = None
        self._unique_classes = None
        self._classes_to_index = {}
        self._probability_vector = None
        self._classes_to_index = None
        self._dset_validator = DatasetValidation()
        self._param_validator = ParameterValidator()

    def _compute_class_probability (self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for the labels in ``X``.

        The label vector is assumed to be stored in the last column of ``X``.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features + 1)
            Training matrix including the ground-truth labels in the last
            column.

        Returns
        -------
        numpy.ndarray
            Array of class probabilities derived from label counts.

        Notes
        -----
        This method returns the probabilities in the order produced by
        ``numpy.unique``.
        """
        labels, label_counts = np.unique(X, return_counts=True)
        probability_vector = None

        if self._unique_classes is None:
            self._unique_classes = np.asarray(labels)
            probability_vector = np.zeros(len(labels), dtype=np.float32)

            for index, label in enumerate(labels):
                self._classes_to_index.update({label: index})
        else:
            probability_vector = np.zeros(len(self._unique_classes))

        for label, label_count in zip(labels, label_counts):
            if label in self._classes_to_index.keys():
                probability_vector[self._classes_to_index.get(label)] = label_count / len(X)
    
    def _compute_impurity (self, X: np.ndarray) -> np.float32:
        """Compute the Gini impurity for a dataset.

        Uses the definition:

        .. math::
            1 - Σ(pᵢ²)

        Parameters
        ----------
        X : numpy.ndarray
            Dataset (or subset) including labels in the last column.

        Returns
        -------
        numpy.float32
            Gini impurity.
        """
        unique_probabilities = self._compute_class_probability(X)
        computed_impurity = 1 - np.sum(np.square(unique_probabilities))
        return computed_impurity
    
    def _compute_entropy (self, X: np.ndarray) -> np.float32:
        """Compute the Shannon entropy for a dataset.

        Uses the definition:

        .. math::
            H(X) = -Σ p(xᵢ) * log₂(p(xᵢ))

        Parameters
        ----------
        X : numpy.ndarray
            Dataset (or subset) including labels in the last column.

        Returns
        -------
        numpy.float32
            Entropy.
        """
        probability_vector = self._compute_class_probability(X)
        probability_vector = probability_vector[probability_vector > 0.0]
        computed_entropy = -(np.sum(probability_vector * np.log2(probability_vector)))
        return computed_entropy
    
    def _compute_log_loss (self, Y_true: np.ndarray, Y_probabilities: np.ndarray) -> np.float32:
        """Compute (binary) log loss.

        Parameters
        ----------
        Y_true : numpy.ndarray of shape (n_samples,)
            Ground-truth labels.
        Y_probabilities : numpy.ndarray of shape (n_samples,)
            Predicted probability for the positive class.

        Returns
        -------
        numpy.float32
            Mean log loss.

        Warnings
        --------
        This function is marked as unstable in the surrounding code and is not
        currently used as a supported split metric.
        """
        log_loss_eq = (Y_true * np.log(Y_probabilities)) + (1 - Y_true) * np.log(1 - Y_probabilities)
        log_loss = 1 / len(Y_probabilities) * np.sum(log_loss_eq)
        return log_loss
    
    def _compute_information_gain (self, X: np.ndarray, left_subset: np.ndarray, right_subset: np.ndarray):
        """Compute information gain for a candidate split.

        Information gain is computed as the impurity of the parent node minus the
        weighted impurities of the child nodes.

        Parameters
        ----------
        X : numpy.ndarray
            Parent dataset (includes labels in the last column).
        left_subset : numpy.ndarray
            Left child dataset.
        right_subset : numpy.ndarray
            Right child dataset.

        Returns
        -------
        numpy.float32
            Information gain.
        """
        main_data_impurity = self._determine_impurity_metric(X)
        left_subset_impurity = self._determine_impurity_metric(left_subset)
        right_subset_impurity = self._determine_impurity_metric(right_subset)

        left_subset_weighted = len(left_subset) / len(X) * left_subset_impurity
        right_subset_weighted = len(right_subset) / len(X) * right_subset_impurity
        
        return main_data_impurity - (left_subset_weighted + right_subset_weighted)
    
    def _determine_impurity_metric (self, X: np.ndarray):
        """Select and compute the configured impurity metric.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset (or subset) including labels in the last column.

        Returns
        -------
        numpy.float32
            Metric score for ``split_metric``.
        """
        if self.split_metric == "gini":
            return self._compute_impurity(X)
        elif self.split_metric == "entropy":
            return self._compute_entropy(X)
        else:
            return self._compute_log_loss() # WARNING: Very volatile code. Don't be stupid and run this line ;)
    
    def _determine_feature_split_metric (self, feature_list: list[int], local_rng: np.random.Generator) -> np.ndarray:
        """Choose which feature indices to consider for splitting.

        Parameters
        ----------
        X : numpy.ndarray
            Input dataset (or feature subset). The last column is assumed to be
            the label column when ``X`` is the full training matrix.

        Returns
        -------
        numpy.ndarray
            Selected feature indices.

        Notes
        -----
        The selection strategy is controlled by the estimator's configuration.
        The current implementation special-cases the string values ``'sqrt'``
        and ``'log2'``.
        """
        if feature_list is None:
            return None

        if self.max_features == "sqrt":
            feature_indices = local_rng.random(
                feature_list,
                size=int(np.sqrt(len(feature_list)))
            )
        elif self.max_features == "log2":
            feature_indices = local_rng.random(
                feature_list,
                size=int(np.log2(len(feature_list)))
            )
        else:
            return feature_list

        return feature_indices
    
    def _split_yield (self, X: np.ndarray, numeric_features: list[int], categorical_features: list[int]):
        """Generate candidate splits for numerical and categorical features.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset to split (includes labels in the last column).
        numeric_features : list of int
            Indices of numerical features to consider.
        categorical_features : list of int
            Indices of categorical features to consider.

        Yields
        ------
        tuple
            A tuple of the form
            ``(feat_type, feat_index, condition, left_subset, right_subset)``.
            ``feat_type`` is either ``'numerical'`` or ``'categorical'``.
        """
        if numeric_features is not None:
            for num_feat in numeric_features:
                unique_values = np.unique(X[:, num_feat])
                computed_midpoint_thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                for threshold in computed_midpoint_thresholds:
                    yield (
                        "numerical",
                        num_feat,
                        threshold,
                        X[X[:, num_feat] > threshold],
                        X[X[:, num_feat] <= threshold]
                    )
        if categorical_features is not None:
            for cat_feat in categorical_features:
                for unique_feat in np.unique(X[:, cat_feat]):
                    yield (
                        "categorical",
                        cat_feat,
                        unique_feat,
                        X[X[:, cat_feat] == unique_feat],
                        X[X[:, cat_feat] != unique_feat]
                    )

    def _split_data (self, X: np.ndarray):
        """Prepare feature subsets and stream candidate splits.

        This method acts as a "middleman" between the recursive tree builder
        (``_build_decision_tree``) and the split generator (``_split_yield``).
        It prepares the dataset for splitting by:

        - Separating numerical and categorical feature subsets (when
            ``categorical_features`` is provided).
        - Determining how many/which feature indices should be considered for
            this split (feature subsampling) via ``_determine_feature_split_metric``.
        - Delegating to ``_split_yield`` to generate candidate split rules and
            child subsets.

        Parameters
        ----------
        X : numpy.ndarray
                Dataset (or subset) to split. During training this includes the
                labels in the last column.

        Yields
        ------
        tuple
                A tuple of the form
                ``(feat_type, feat_index, condition, left_subset, right_subset)``
                which is forwarded to ``_build_decision_tree``.
        """
        yield from self._split_yield(
            X,
            self._determine_feature_split_metric(self.numerical_features),
            self._determine_feature_split_metric(self.categorical_features)
        )

    def _create_node (
        self,
        split_index: int = None,
        split_num_condition: Union[int, float] = None,
        split_cat_condition: Union[int, float] = None,
        information_gain: float = None,
        computed_class_probabilities: np.ndarray = None,
        create_decision_node: bool = False,
        create_leaf_node: bool = False,
    ) -> Union[DecisionNode, LeafNode]:
        """Create and return a decision node or leaf node.

        Parameters
        ----------
        split_index : int
            Feature index used for splitting.
        split_num_condition : int or float or None
            Numerical threshold used for the split.
        split_cat_condition : int or float or None
            Categorical equality condition used for the split.
        information_gain : float
            Information gain of the chosen split.
        computed_class_probabilities : numpy.ndarray
            Class distribution to store in a leaf node.
        create_decision_node : bool, default=False
            If True, create a :class:`DecisionNode`.
        create_leaf_node : bool, default=False
            If True, create a :class:`LeafNode`.

        Returns
        -------
        DecisionNode or LeafNode
            Instantiated node.
        """
        if create_decision_node:
            if split_num_condition is not None:
                instantiated_decision_node = DecisionNode(
                    split_index=split_index,
                    split_feat_num_condition=split_num_condition,
                    split_feat_cat_condition=None,
                    information_gain=information_gain
                )
            if split_cat_condition is not None:
                instantiated_decision_node = DecisionNode(
                    split_index=split_index,
                    split_feat_num_condition=None,
                    split_feat_cat_condition=split_cat_condition,
                    information_gain=information_gain
                )
            return instantiated_decision_node

        if create_leaf_node:
            instantiated_leaf_node = LeafNode(computed_class_probabilities)
            instantiated_leaf_node._node_is_leaf = True
            return instantiated_leaf_node

    def _build_decision_tree (self, X: np.ndarray, recursive_tree_depth: int = 1):
        """Recursively build the decision tree.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset (or subset) including labels in the last column.
        recursive_tree_depth : int, default=1
            Current depth in the recursion.

        Returns
        -------
        DecisionNode or LeafNode
            Root node of the (sub)tree.
        """
        best_computed_information_gain = 0
        best_index = None
        best_num_split_condition = None
        best_unique_split_condition = None
        best_computed_left_split: np.ndarray = None
        best_computed_right_split: np.ndarray = None

        # Tree depth & min_samples_split hyperparameter check
        if recursive_tree_depth == self.max_depth:
            print(f"[*] Stopping training, condition hit: Max depth")
            leaf_node = self._create_node(
                computed_class_probabilities=self._compute_class_probability(X), 
                create_leaf_node=True
            )
            return leaf_node
        
        if len(X) <= self.min_samples_split:
            print(f"[*] Stopping training, condition hit: Minimum samples split")
            leaf_node = self._create_node(
                computed_class_probabilities=self._compute_class_probability(X), 
                create_leaf_node=True
            )
            return leaf_node

        for feat_type, feat_index, condition, group_above_condition, group_below_condition in self._split_data(X):
            if feat_type == "numerical":
                num_feat_information_gain = self._compute_information_gain(
                    X, 
                    group_above_condition, 
                    group_below_condition
                )
                if num_feat_information_gain > best_computed_information_gain:
                    best_index = feat_index
                    best_computed_information_gain = num_feat_information_gain
                    best_num_split_condition = condition
                    best_computed_left_split = group_above_condition
                    best_computed_right_split = group_below_condition
                    best_unique_split_condition = None
                else:
                    continue

            if feat_type == "categorical":
                cat_feat_information_gain = self._compute_information_gain(
                    X, 
                    group_above_condition, 
                    group_below_condition
                )

                if cat_feat_information_gain > best_computed_information_gain:
                    best_index = feat_index
                    best_computed_information_gain = cat_feat_information_gain
                    best_unique_split_condition = condition
                    best_computed_left_split = group_above_condition
                    best_computed_right_split = group_below_condition
                    best_num_split_condition = None
                else:
                    continue

            print(f"""
                Information
                    Best index: {best_index}
                    Best computed information gain: {best_computed_information_gain}
                    Best computed left subset: {best_computed_left_split}
                    Best computed right subset: {best_computed_right_split}

                Conditions
                    Best numerical condition: {best_num_split_condition}
                    Best categorical conditon: {best_unique_split_condition}
            """)

        # min_information_gain hyperparameter check
        if best_computed_information_gain < self.min_information_gain:
            print(f"[*] Stopping training, condition hit: Minimum information gain")
            instantiated_leaf_node = self._create_node(
                computed_class_probabilities=self._compute_class_probability(X), 
                create_leaf_node=True
            )
            return instantiated_leaf_node

        # min_samples_leaf hyperparameter check
        if (best_computed_left_split is not None and
            best_computed_right_split is not None
        ):
            if (best_computed_left_split.shape[0] < self.min_samples_leaf or
                best_computed_right_split.shape[0] < self.min_samples_leaf
            ):
                print(f"[*] Stopping training, condition hit: Minimum samples split")
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

        instantiated_decision_node._left_node = self._build_decision_tree(
            best_computed_left_split, 
            recursive_tree_depth + 1
        )
        instantiated_decision_node._right_node = self._build_decision_tree(
            best_computed_right_split, 
            recursive_tree_depth + 1
        )

        return instantiated_decision_node

    def _inference_traversal_tree (self, X: np.ndarray, root_node: Union[DecisionNode, LeafNode]):
        """Traverse the tree for a single sample and return the predicted class.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_features,)
            Feature vector for a single sample.
        root_node : DecisionNode or LeafNode
            Current node during traversal.

        Returns
        -------
        tuple
            ``(is_leaf, predicted_class)`` where ``is_leaf`` is a boolean flag
            and ``predicted_class`` is the class index (argmax at the leaf).
        """
        if root_node._node_is_leaf:
            return True, root_node.compute_argmax()

        if root_node._feat_num_condition is not None:
            if X[root_node._split_index] > root_node._feat_num_condition:
                return self._inference_traversal_tree(X, root_node._left_node)
            else:
                return self._inference_traversal_tree(X, root_node._right_node)
            
        if root_node._feat_cat_condition is not None:
            if X[root_node._split_index] == root_node._feat_cat_condition:
                return self._inference_traversal_tree(X, root_node._left_node)
            else:
                return self._inference_traversal_tree(X, root_node._right_node)

    def fit (self, X: np.ndarray):
        """Fit the decision tree classifier.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features + 1)
            Training data. The last column must contain the class labels.

        Returns
        -------
        None
            This estimator is fitted in-place.
        """
        self._local_rng = np.random.default_rng(self.random_state)
        self._dset_validator.perform_dataset_validation(X)
        self._root_node = self._build_decision_tree(X)

    def predict (self, X: np.ndarray):
        """Predict class labels for samples in ``X``.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted class indices.
        """
        inferenced_elements_array = []

        if self._unique_classes is None:
            raise RuntimeError("[-] Error: The DecisionTreeClassifier has not been fitted yet")

        for row in X:
            leaf_node, predicted_class = self._inference_traversal_tree(row, self._root_node)
            if leaf_node:
                inferenced_elements_array.append(predicted_class)
            
        return np.asarray(inferenced_elements_array)

def main ():
    # --- Import libraries ---
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

    # --- 1. Generate dataset ---
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    # --- 2. Split dataset into training and testing ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Custom tree expects labels in last column during fit
    train_combined = np.column_stack([X_train, y_train])

    # --- 3. Train custom DecisionTreeClassifier ---
    custom_tree = DecisionTreeClassifier(
        split_metric="gini",
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    custom_tree.fit(train_combined)

    # --- 4. Get predictions and compute metrics for custom model ---
    custom_predictions = custom_tree.predict(X_test)
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    custom_precision = precision_score(y_test, custom_predictions, average="macro", zero_division=0)
    custom_recall = recall_score(y_test, custom_predictions, average="macro", zero_division=0)

    # --- 5. Train sklearn DecisionTreeClassifier and compute metrics ---
    sklearn_tree = SklearnDecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    sklearn_tree.fit(X_train, y_train)

    sklearn_predictions = sklearn_tree.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    sklearn_precision = precision_score(y_test, sklearn_predictions, average="macro", zero_division=0)
    sklearn_recall = recall_score(y_test, sklearn_predictions, average="macro", zero_division=0)

    # --- Side-by-side comparison ---
    print("\n" + "=" * 60)
    print("DECISION TREE CLASSIFIER - METRICS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Custom Model':<20} {'Sklearn Model':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {custom_accuracy:<20.4f} {sklearn_accuracy:<20.4f}")
    print(f"{'Precision':<15} {custom_precision:<20.4f} {sklearn_precision:<20.4f}")
    print(f"{'Recall':<15} {custom_recall:<20.4f} {sklearn_recall:<20.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
