#!/bin/bash

# Global constants
PYTHONPATH="/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai"

# Generator compiler paths
DATASET_GENERATOR_VENV="${PYTHONPATH}/python-utilities/generator-venv"
DATASET_GENERATOR_PATH="${PYTHONPATH}/python-utilities/generator-files/generator.py"

# Machine learning algorithms path
LIN_REG_SOURCE="${PYTHONPATH}/linear-regression/source-file/linear-regression.py"
LOG_REG_SOURCE="${PYTHONPATH}/logistic-regression/source-files/logistic-regression.py"
TREE_SOURCE="${PYTHONPATH}/decision-tree/source-code/tree.py"

# Global Top Level Domain and VENV checks
main_tld_passed=0
main_venv_passed=0

function check_tld_venv () {
    if [[ ${main_tld_passed} -eq 1 && ${main_venv_passed} -eq 1 ]]; then
        echo "[+] PYTHON_TLD and VIRTUAL ENVIRONMENT IS SET"
    else
        # Set the PYTHON_TLD
        {
            export PYTHONPATH="${PYTHONPATH}" &&
            echo "[+] PYTHON_TLD is now set: $(echo $PYTHONPATH)" &&
            main_tld_passed=1;
        } || {
            echo "[-] Unable to set PYTHON_TLD" &&
            exit 1;
        }

        # Set the Python virtual environment
        {
            source "main-venv/bin/activate" && 
            echo "[+] VENV is now set!" &&
            main_venv_passed=1;
        } || {
            echo "[-] Unable to set VENV" &&
            exit 1;
        }
        check_tld_venv
    fi
}

function generate_datasets () {
    local algorithm_type="$1"
    local key_value_change="$2"
    local valid_algorithms=("regression" "classification" "clustering")

    if [[ ${main_tld_passed} -eq 1 && ${main_venv_passed} -eq 1 ]]; then
        for algorithm in "${valid_algorithms[@]}"; do
            if [[ ! "${algorithm_type}" == "${algorithm}" ]]; then
                echo "[-] Error: Invalid algorithm type passed"
                exit 1
            fi
        done

        python3 "${DATASET_GENERATOR_PATH}" \
            --dataset-type "${algorithm_type}" \
            --key-value-change "${key_value_change}"
    fi
}

function execute_model () {
    local algorithm_to_use="$1"
    declare -A machine_learning_algorithms=(
        ["linreg"]="${LIN_REG_SOURCE}"
        ["logreg"]="${LOG_REG_SOURCE}"
        ["tree"]="${TREE_SOURCE}"
    )

    for algorithm in "${!machine_learning_algorithms[@]}"; do
        algorithm_source="${machine_learning_algorithms[${algorithm}]}"
        if [[ "${algorithm}" == "${algorithm_to_use}" ]]; then
            echo "[+] Running: ${algorithm_source}"
            python3 "${algorithm_source}"
        fi
    done
}

function main () {
    check_tld_venv
    local mode="$1"
    local algorithm_type="$2"
    local dataset_type="$3"
    local key_value_change="$4"

    if [[ "${mode}" == "dataset" ]]; then
        echo "[+] Generating dataset following ML algorithm type: ${dataset_type}"
        generate_datasets "${dataset_type}" "${key_value_change}"
    elif [[ "${mode}" == "algorithm" ]]; then
        echo "[+] Running machine learning algorithm: ${algorithm_type}"
        execute_model "${algorithm_type}"
    else
        echo "[-] Unrecognized option"
        exit 1
    fi
}

main "$@"
