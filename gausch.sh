#!/bin/bash


module purge
source ~/git_projects/jlkiams/TSCO/env_python.sh
schnetpath="gausch.py"

molinputpath=`date +'%M%S%N' | md5sum | tr -d ' -'`

read atoms derivs charge spin < $2

#Create input file
echo $(sed -n 2,$(($atoms+1))p < $2 | cut -c 1-72) > /dev/shm/${molinputpath}.txt

### Run schnet predictor
echo "Running schnet predictor..."
python $schnetpath $atoms $derivs $3 $molinputpath

rm /dev/shm/${molinputpath}.*

