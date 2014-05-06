#!/bin/bash

set -e

ROOT=$(dirname $(readlink -e $0))
LOCAL_MACHINE_SCRATCH_SPACE=/home/scratch
ENV="$(mktemp -u -d -p "$LOCAL_MACHINE_SCRATCH_SPACE" "conda_env.$USER.XXXXXXXXXXXX")/conda"

function safe_call {
    # usage:
    #   safe_call function param1 param2 ...

    HERE=$(pwd)
    "$@"
    cd "$HERE"
}

function conda_install {
    conda install --yes "$1"
}

function pip_install {
    pip install "$1"
}

conda create --yes --prefix "$ENV" python
source activate "$ENV"

echo "$ENV" > "$ROOT/.env"

safe_call conda_install numpy
safe_call conda_install scipy
