#!/bin/bash

# Global constants
PYTHON_TLD=$(pwd)
LIN_REG_SOURCE="/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai/linear-regression/source-file/lin-reg-new.py"
LOG_REG_SOURCE="/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai/logistic-regression/source-files/log-reg-main.py"

tld_passed=0
venv_passed=0

function check_tld () {
    if [[ "${PYTHONPATH}" == "${PYTHON_TLD}" ]] ; then
        echo "[+] Python TLD is set"
        tld_passed=1
    else
        echo "[-] Error: Python TLD is not set. Setting now"
        export PYTHONPATH=${PYTHON_TLD} && echo "[+] Python TLD set: ${PYTHON_TLD}"
        check_tld
    fi
}

function check_venv () {
    if [[ -n "$VIRTUAL_ENV" ]] ; then
        echo "[+] Python virtual environment is set"
        venv_passed=1
    else
        echo "[-] Error: Python virtual environment isn't set. Setting now"
        source main-venv/bin/activate && echo "[+] Environment set: ${VIRTUAL_ENV}"
        check_venv
    fi
}

function main () {
  check_tld
  check_venv

  if [[ ${tld_passed} -eq 1 && ${venv_passed} -eq 1 ]] ; then
      echo "[+] TLD and VENV set. Running: $1"
      if [[ $1 == "linreg" ]] ; then
          echo "[+] Running Linear Regression model"
          python3 ${LIN_REG_SOURCE}
      elif [[ "$1" == "logreg" ]] ; then
          :
      fi
  fi
}

main $1

