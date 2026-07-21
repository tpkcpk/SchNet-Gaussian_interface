#!/bin/bash
source ~/script/python_env.sh

### Run schnet predictor
echo "Running schnet predictor..."
python gausch.py $@
