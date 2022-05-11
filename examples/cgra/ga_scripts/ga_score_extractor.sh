#!/bin/bash

typeset -a results
if [[ $# -lt 2 ]]; then
	echo "Usage: $0 <target data file> <stdout files>"
fi

output=$1
echo "" -n > $output
shift
while [[ $# -gt 0 ]]; do
	input=$1
	shift

	# Get the second to last line, which has the II for the graph.
	ii=$(tail -n 2 $input | head -n 1 | cut -f 7 -d' ' )

	echo "$ii, " >> $output
done
