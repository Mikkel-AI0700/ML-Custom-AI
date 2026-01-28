import os
import re
import json
import argparse
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    make_regression,
    make_classification,
    make_blobs
)

def _read_generator_configuration (json_configuration_path: Path):
    """Read a JSON generator configuration file.

    Parameters
    ----------
    json_configuration_path : pathlib.Path
        Path to a JSON file containing keyword arguments for a scikit-learn
        dataset generator.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with json_configuration_path.open("r") as config_path:
        dataset_configuration_dict = json.load(config_path)
    return dataset_configuration_dict

def _write_to_file (filepath: Path, X: pd.DataFrame, Y: pd.DataFrame):
    """Split `X`/`Y` into train/test and write them as CSV files.

    Parameters
    ----------
    filepath : pathlib.Path
        Output directory where the CSV files will be written.
    X : pandas.DataFrame
        Feature dataset.
    Y : pandas.DataFrame
        Target/label dataset.

    Returns
    -------
    None

    Notes
    -----
    This function prints a message for files matching the pattern
    ``(train|test)_(x|y)_dataset.csv`` in the output directory, but it does not
    delete those files.
    """
    for directory_content in os.listdir(filepath):
        if re.findall(r"(train|test)_(x|y)_dataset.csv", directory_content):
            print(f"[*] Removed: {directory_content}")

    train_x, test_x, train_y, test_y = train_test_split(
        X,
        Y,
        train_size=0.8,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    generated_datasets = [train_x, train_y, test_x, test_y]
    filenames = ["train_x.csv", "train_y.csv", "test_x.csv", "test_y.csv"]

    for (dataset, data_filename) in zip(generated_datasets, filenames):
        dataset = pd.DataFrame(dataset)
        dataset.to_csv(os.path.join(filepath, data_filename), index=False)

def _change_configuration (config_key_value: str, generator_configuration: dict[str, Any]):
    """Update a generator configuration dictionary from a key/value string.

    Parameters
    ----------
    config_key_value : str
        Comma-separated list of assignments, e.g. ``"n_samples=200, n_features=5"``.
        If `None`, no changes are applied.
    generator_configuration : dict[str, Any]
        Configuration dictionary to update.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Exits with status code ``1`` if an unknown configuration key is
        provided.
    """
    try:
        if config_key_value is None:
            return

        for config_key in config_key_value.split(","):
            stripped_config_key = config_key.strip()
            config_name, config_value = stripped_config_key.split("=", 1)

            if config_name in generator_configuration.keys():
                generator_configuration.update({config_name: config_value})
            else:
                raise ValueError("[-] Error: Non existent configuration key detected.")
    except ValueError as non_existent_config_key:
        print(non_existent_config_key)
        exit(1)

def create_regression (
    dataset_gen_path: Path, 
    regressor_config: Path, 
    key_value_config: dict[str, Any]
):
    """Generate a regression dataset and write train/test CSV files.

    Parameters
    ----------
    dataset_gen_path : pathlib.Path
        Output directory for generated CSV files.
    regressor_config : pathlib.Path
        JSON configuration path for `sklearn.datasets.make_regression`.
    key_value_config : str or None
        Optional key/value override string passed to `_change_configuration`.

    Returns
    -------
    None
    """
    regressor_configuration = _read_generator_configuration(regressor_config)
    _change_configuration(key_value_config, regressor_configuration)
    X, Y = make_regression(**regressor_configuration)
    _write_to_file(dataset_gen_path, pd.DataFrame(X), pd.DataFrame(Y))

def create_classification (
    dataset_gen_path: Path, 
    classification_config: Path, 
    key_value_config: dict[str, Any]
):
    """Generate a classification dataset and write train/test CSV files.

    Parameters
    ----------
    dataset_gen_path : pathlib.Path
        Output directory for generated CSV files.
    classification_config : pathlib.Path
        JSON configuration path for `sklearn.datasets.make_classification`.
    key_value_config : str or None
        Optional key/value override string passed to `_change_configuration`.

    Returns
    -------
    None
    """
    classification_configuration = _read_generator_configuration(classification_config)
    _change_configuration(key_value_config, classification_configuration)
    X, Y = make_classification(**classification_configuration)
    _write_to_file(dataset_gen_path, pd.DataFrame(X), pd.DataFrame(Y))

def create_clustering (
    dataset_gen_path: Path, 
    clustering_config: Path, 
    key_value_config: dict[str, Any]
):
    """Generate a clustering dataset and write train/test CSV files.

    Parameters
    ----------
    dataset_gen_path : pathlib.Path
        Output directory for generated CSV files.
    clustering_config : pathlib.Path
        JSON configuration path for `sklearn.datasets.make_blobs`.
    key_value_config : str or None
        Optional key/value override string passed to `_change_configuration`.

    Returns
    -------
    None
    """
    clustering_configuration = _read_generator_configuration(clustering_config)
    _change_configuration(key_value_config, clustering_configuration)
    X, Y = make_blobs(**clustering_configuration)
    _write_to_file(dataset_gen_path, pd.DataFrame(X), pd.DataFrame(Y))


def main ():
    """Entry point for the dataset generator CLI.

    Parses command-line arguments and generates datasets in one of three modes:
    regression, classification, or clustering.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Exits with status code ``1`` on invalid arguments.
    """
    BASE_PATH = Path(__file__).resolve().parent.parent

    regressor_dataset_path = BASE_PATH / "test-data/regression-data"
    classification_dataset_path = BASE_PATH / "test-data/classification-data"
    clustering_dataset_path = BASE_PATH / "test-data/clustering-data"

    regressor_json_path = BASE_PATH / "json-config-files/regressor/regressor-generator-configuration.json"
    classification_json_path = BASE_PATH / "json-config-files/classification/classification-generator-configuration.json"
    clustering_json_path = BASE_PATH / "json-config-files/clustering/clustering-generator-configuration.json"

    argp = argparse.ArgumentParser(description="ML dataset generators")
    argp.add_argument("--dataset-type", required=True, dest="dset_type")
    argp.add_argument("--use-default", required=False, action="store_true")
    argp.add_argument("--key-value-change", required=False, dest="key_value_parameter")

    parsed_arguments = argp.parse_args()

    try:
        if parsed_arguments.dset_type == "regression":
            print("[+] Creating training and testing regression datasets")
            create_regression(
                regressor_dataset_path,
                regressor_json_path, 
                parsed_arguments.key_value_parameter
            )
        elif parsed_arguments.dset_type == "classification":
            print("[+] Creating training and testing classification datasets")
            create_classification(
                classification_dataset_path,
                classification_json_path,
                parsed_arguments.key_value_change
            )
        elif parsed_arguments.dset_type == "clustering":
            print("[+] Creating training and testing clustering datasets")
            create_clustering(
                clustering_dataset_path,
                clustering_json_path,
                parsed_arguments.key_value_change
            )
        else:
            raise ValueError(f"[-] Error: Incorrect dataset type -> {parsed_arguments.dset_type}")
    except ValueError as incorrect_argument_error:
        print(incorrect_argument_error)
        exit(1)

main()
