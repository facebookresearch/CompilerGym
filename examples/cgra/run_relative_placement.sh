#!/bin/bash

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <model path>"
	echo "Use the --train flag to train (on the python script)"
	exit 1
fi

rm -f relative_placement_output/out
mkdir -p relative_placement_output
echo "Starting program"
python relative_placement_model.py --number 10000 --test $1 &> relative_placement_output/out
