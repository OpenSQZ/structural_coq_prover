#!/bin/bash
# Source this file to set PYTHONPATH for the project
# Usage: source init_env.sh

export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"