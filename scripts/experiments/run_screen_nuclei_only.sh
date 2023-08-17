#!/bin/bash

cd "$(dirname "$0")"
cd ../..
for file in config/image_embedding/other/screen_nuclei_only/cv/fold_3/*; do python run.py --config $file; done