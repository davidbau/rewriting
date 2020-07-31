#!/usr/bin/env bash

# This script creates directories of images that
# reproduce the watermark experiment from section
# 5.2 of the paper.

set -ex
# Start from directory of script
cd "$(dirname "$(readlink -f "$0")")"

python -m metrics.make_watermark_images \
   --erasemethod ours --nreps 2 --drank 60 --rank 1

python -m metrics.make_watermark_images \
   --erasemethod ours --nreps 2 --drank 30 --rank 1

python -m metrics.make_watermark_images \
   --erasemethod gandissect --drank 30

python -m metrics.make_watermark_images \
   --erasemethod gandissect --drank 60

python -m metrics.make_watermark_images \
   --erasemethod none
