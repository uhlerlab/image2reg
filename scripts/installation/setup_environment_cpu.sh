#!/bin/bash

cd "$(dirname "$0")"
cd ../..

for file in requirements/cpu/*; do pip install -r $file; done