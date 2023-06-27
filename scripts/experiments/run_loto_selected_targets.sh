#!/bin/bash

cd "$(dirname "$0")"
cd ../..
for file in config/image_embedding/specific_targets/loto/train_resnet/new/*; do python run.py --config $file; done
