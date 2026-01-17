#!/bin/bash

# Global constants
PYTHON_TLD=$(pwd)

# Generator compiler paths
DATASET_GENERATOR_VENV="${PYTHON_TLD}/python-utilities/generator-venv"
DATASET_GENERATOR_PATH="${PYTHON_TLD}/python-utilities/generator-files/generator.py"

# Machine learning algorithms path
LIN_REG_SOURCE="${PYTHON_TLD}/linear-regression/source-file/lin-reg-new.py"
LOG_REG_SOURCE="${PYTHON_TLD}/logistic-regression/source-files/log-reg-new.py"
TREE_SOURCE=""

# Global Top Level Domain and VENV checks
main_tld_passed=0
main_venv_passed=0

function check_tld_venv () {
    if [[ ${main_tld_passed} -eq 1 && ${main_venv_passed} -eq 1 ]]; then
        echo "[+] PYTHON_TLD and VIRTUAL ENVIRONMENT IS SET"
    else
        { export PYTHON_TLD="${PYTHON_TLD}" && echo "[+] PYTHON_TLD is now set!" && main_tld_passed=1; } || { echo "[-] Unable to set PYTHON_TLD" && exit 1; }
        { source "main-venv/bin/activate" && echo "[+] VENV is now set!" && main_venv_passed=1; } || { echo "[-] Unable to set VENV" && exit 1; }
        check_tld_venv
    fi
}

function generate_datasets () {
    local algorithm_type="$1"
    local key_value_change="$2"

    if [[ ${gen_tld_passed} -eq 1 && ${gen_venv_passed} -eq 1 ]]; then
        if [[ "${algorithm_type}" == "regression" ]]; then
            :
        elif [[ "${algorithm_type}" == "classification" ]]; then
            :
        elif [[ "${algorithm_type}" == "clustering" ]]; then
            :
        else
            :
        fi
    fi
}

function main () {
    check_tld_venv

    if [[ ${tld_passed} -eq 1 && ${venv_passed} -eq 1 ]] ; then
        echo "[+] TLD and VENV set. Running: $1"
        if [[ $1 == "linreg" ]] ; then
            echo "[+] Running Linear Regression model"
            python3 ${LIN_REG_SOURCE}
        elif [[ "$1" == "logreg" ]] ; then
            python3 ${LOG_REG_SOURCE}
        elif [[ "$1" == "tree" ]]; then
            :
        fi
    fi
}

main $1

