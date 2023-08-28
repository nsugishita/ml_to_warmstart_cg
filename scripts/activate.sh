#!/bin/sh

# Initialise env vars.
# Call this script from the top directory of this project.
# . scripts/activate.sh

export PYTHONPATH="$PYTHONPATH":"$(pwd)/extensions/uniquelist/build"
export OMP_NUM_THREADS=1

. ./env/bin/activate
