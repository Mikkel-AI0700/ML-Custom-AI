#!/bin/bash

# Global constants
PYTHON_TLD="$(pwd)"
LIN_REG_SOURCE = "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai/linear-regression/source-file/lin-reg-source.py"
LOG_REG_SOURCE = "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai/logistic-regression/source-files/log-reg-main.py"

function check_tld () {
    if [[ $PYTHONPATH == ${PYTHON_TLD} ]] ; then
        echo "[+] Python TLD is set"
        echo "true"
    else
        echo "[-] Error: Python TLD is not set. Setting now"
        export PYTHONPATH=${PYTHON_TLD}
    fi
}

function check_venv () {
    if [[ -n $VIRTUAL_ENV ]] ; then
        echo "[+] Python virtual environment is set"
        echo "true"
    else
        echo "[-] Error: Python virtual environment isn't set. Setting now"
        source main-venv/bin/activate
    fi
}

function main () {
    if [[ ${check_tld} == "true"]] && [[ ${check_venv} == "true" ]] ; then
        if [[ $1 == "lin" ]] || [[ $1 == "lin-reg" ]] ; then
            python3 ${LIN_REG_SOURCE}
        else if [[ $1 == "log" ]] || [[ $1 == "log-reg" ]] ; then
            python3 ${LOG_REG_SOURCE}
        else
            echo -e "[-] Error: User passed argument doesn't exist"
            exit 1
        fi
    fi
}

main

