#!/bin/bash

cd "$(dirname "$0")"
cd ../..
for file in config/profile_embedding/screen/cv/fold_3/*; do python run.py --config $file; done