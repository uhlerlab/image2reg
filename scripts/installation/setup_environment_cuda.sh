#!/bin/bash

cd "$(dirname "$0")"
cd ../..

for file in requirements/cuda/*; do pip install -r $file; done