#!/usr/bin/env bash

# Bash script to set up an anaconda python-based deep learning environment
# that has support for pytorch and needed dependencies.  It assumes that
# you have (mini)conda installed.

# This should not require root.  However, it does copy and build a lot of
# binaries into your ~/.conda directory.  If you do not want to store
# these in your homedir disk, then ~/.conda can be a symlink somewhere else.
# (At MIT CSAIL, you should symlink ~/.conda to a directory on NFS or local
# disk instead of leaving it on AFS, or else you will exhaust your quota.)

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# Default RECIPE 'renv' can be overridden by 'RECIPE=foo setup.sh'
RECIPE=${RECIPE:-renv}
# Default ENV_NAME 'renv' can be overridden by 'ENV_NAME=foo setup.sh'
ENV_NAME="${ENV_NAME:-${RECIPE}}"
echo "Creating conda environment ${ENV_NAME}"

if [[ ! $(type -P conda) ]]
then
    echo "conda not in PATH"
    echo "read: https://conda.io/docs/user-guide/install/index.html"
    exit 1
fi

if df "${HOME}/.conda" --type=afs > /dev/null 2>&1
then
    echo "Not installing: your ~/.conda directory is on AFS."
    echo "Use 'ln -s /some/nfs/dir ~/.conda' to avoid using up your AFS quota."
    exit 1
fi

# Uninstall existing environment
source deactivate
rm -rf ~/.conda/envs/${ENV_NAME}

# Build new environment based on the recipe.
conda env create --name=${ENV_NAME} -f setup/${RECIPE}.yml

# Set up CUDA_HOME to set itself up correctly on every source activate
# https://stackoverflow.com/questions/31598963
mkdir -p ~/.conda/envs/${ENV_NAME}/etc/conda/activate.d
echo "export CUDA_HOME=/usr/local/cuda-10.1" \
    > ~/.conda/envs/${ENV_NAME}/etc/conda/activate.d/CUDA_HOME.sh

source activate ${ENV_NAME}
