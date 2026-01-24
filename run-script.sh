#!/bin/bash

# Global constants
PYTHON_TLD=$(pwd)

# Generator compiler paths
DATASET_GENERATOR_VENV="${PYTHON_TLD}/python-utilities/generator-venv"
DATASET_GENERATOR_PATH="${PYTHON_TLD}/python-utilities/generator-files/generator.py"

# Machine learning algorithms path
LIN_REG_SOURCE="${PYTHON_TLD}/linear-regression/source-file/lin-reg-new.py"
LOG_REG_SOURCE="${PYTHON_TLD}/logistic-regression/source-files/log-reg-new.py"
TREE_SOURCE="${PYTHON_TLD}/decision-tree/source-code/tree.py"

# Global Top Level Domain and VENV checks
main_tld_passed=0
main_venv_passed=0

function check_tld_venv () {
    if [[ ${main_tld_passed} -eq 1 && ${main_venv_passed} -eq 1 ]]; then
        echo "[+] PYTHON_TLD and VIRTUAL ENVIRONMENT IS SET"
    else
        check_tld_venv

        # Set the PYTHON_TLD
        {
            export PYTHON_TLD="${PYTHON_TLD}" &&
            echo "[+] PYTHON_TLD is now set!" &&
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
    fi
}

function generate_datasets () {
    local algorithm_type="$1"
    local key_value_change="$2"
    local valid_algorithms=("regression" "classification" "clustering")

    if [[ ${main_tld_passed} -eq 1 && ${main_venv_passed} -eq 1 ]]; then
        for algorithm in "${valid_algorithms[@]}"; do
            if [[ ! "${algorithm_type}" != "${algorithm}" ]]; then
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
        generate_datasets "${dataset_type}" "${key_value_change}"
    elif [[ "${mode}" == "algorithm" ]]; then
        execute_model "${algorithm_type}"
    else
        echo "[-] Unrecognized option"
        exit 1
    fi
}

main $@

