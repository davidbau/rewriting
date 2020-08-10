#!/usr/bin/env bash

# This script creates directories of images that
# reproduce the watermark experiment from section
# 5.2 of the paper.

set -ex
# Start from directory of script
cd "$(dirname "$(readlink -f "$0")")"

if [ -f "results/kitchen/layer4/netpqc/1000/labels.json" ]
then
    echo Kitchen dissection already downloaded
else
    mkdir -p results
    pushd results
    wget https://rewriting.csail.mit.edu/data/dissection/kitchen.zip
    unzip kitchen.zip
    rm kitchen.zip
    popd
fi
echo Ready to run notebooks/reflection-rule-change.ipynb
