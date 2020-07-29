#!/usr/bin/env bash

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

