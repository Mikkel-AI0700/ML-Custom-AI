#!/bin/bash

# Global constants
PYTHON_TLD="$(pwd)"
LIN_REG_SOURCE="/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai/linear-regression/source-file/lin-reg-source.py"
LOG_REG_SOURCE="/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai/logistic-regression/source-files/log-reg-main.py"

tld_passed=false
venv_passed=false

function check_tld () {
    if [[ "${PYTHONPATH}" == "${PYTHON_TLD}" ]] ; then
        echo "[+] Python TLD is set"
        tld_passed=true
    else
        echo "[-] Error: Python TLD is not set. Setting now"
        export PYTHONPATH=${PYTHON_TLD}
    fi
}

function check_venv () {
    if [[ -n "$VIRTUAL_ENV" ]] ; then
        echo "[+] Python virtual environment is set"
        venv_passed=true
    else
        echo "[-] Error: Python virtual environment isn't set. Setting now"
        source main-venv/bin/activate
    fi
}

function main () {
  if [[ ${tld_passed} == true && ${venv_passed} == true ]] ; then
      if [[ "$1" == "linreg" ]] ; then
          python3 ${LIN_REG_SOURCE}
      elif [[ "$1" == "logreg" ]] ; then
          :
      fi
  fi
}

main $1

