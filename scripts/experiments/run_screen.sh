#!/bin/bash

cd "$(dirname "$0")"
cd ../..
for file in config/image_embedding/screen/cv_new/fold_0/*; do python run.py --config $file; done