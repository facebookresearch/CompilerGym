#!/bin/bash

rm -rf ga_output
mkdir -p ga_output
parallel 'echo "Starting iter {}"; python ga.py --nomp --max_cands 100 --env=cgra-v0 --benchmark=dfg_10/{} --reward=II &> ga_output/out_{}' ::: $(seq 0 10000)
