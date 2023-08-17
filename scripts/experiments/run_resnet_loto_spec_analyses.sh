#!/bin/bash

cd "$(dirname "$0")"
cd ../..
for file in config/image_embedding/specific_targets/loto/analyze_loto_exp/resnet_analyses/*; do python run.py --config $file; done