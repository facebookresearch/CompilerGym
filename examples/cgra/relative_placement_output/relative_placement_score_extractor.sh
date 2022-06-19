#!/bin/bash

typeset -a results
if [[ $# -ne 2 ]]; then
    echo "Usage $0 <target data file> <stdout file>"
fi

output=$2
input=$1
if [[ -f $output ]]; then
    echo "Output $output already exists (will be overwritten, please manually delete)"
    exit 1
fi
results=( $(grep -e 'Exectuted' -B 1 $input | grep -e 'Computing Reward' | cut -f7 -d' ') )

for r in ${results[@]}; do
    echo "$r, " >> $output
done