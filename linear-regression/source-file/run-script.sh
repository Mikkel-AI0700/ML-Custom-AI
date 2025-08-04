#!/bin/bash

PYTHON_TLD="/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai/linear-regression/"

if [[ "$PYTHONPATH" == ${PYTHON_TLD} ]] ; then
        echo -e "[+] Python TLD is set!"
else
    echo -e "[-] Python TLD is not set. Setting now!"
    export PYTHONPATH=${PYTHON_TLD} && \
        echo -e "[+] Successfully set Python TLD: $(echo "$PYTHONPATH")"
fi

if [[ -n "$VIRTUAL_ENV" ]] ; then
        echo -e "[+] Python virtual environment is set!"
        python3 lin-reg-source-copy.py
else
    echo -e "[-] Python virtual environment isn't set. Setting now!"
    source ../../main-venv/bin/activate && python3 lin-reg-source-copy.py
fi

