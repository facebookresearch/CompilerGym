#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -ve
mkdir -p results

python run_loop_tool_sweep.py  1 4 1 | tee results/s1_v4_log.txt
python run_loop_tool_sweep.py  1 1 1 | tee results/s1_v1_log.txt

python run_loop_tool_sweep.py  2 4 1 | tee results/s2_v4_log.txt
python run_loop_tool_sweep.py  2 1 1 | tee results/s2_v1_log.txt

python run_loop_tool_sweep.py  4 4 1 | tee results/s4_v4_log.txt
python run_loop_tool_sweep.py  4 1 1 | tee results/s4_v1_log.txt

python run_loop_tool_sweep.py  8 4 1 | tee results/s8_v4_log.txt
python run_loop_tool_sweep.py  8 1 1 | tee results/s8_v1_log.txt

python run_loop_tool_sweep.py  16 4 1 | tee results/s16_v4_log.txt
python run_loop_tool_sweep.py  16 1 1 | tee results/s16_v1_log.txt

python run_loop_tool_sweep.py  1 4 0 | tee results/s1_v4_linear.txt
python run_loop_tool_sweep.py  1 1 0 | tee results/s1_v1_linear.txt

python run_loop_tool_sweep.py  2 4 0 | tee results/s2_v4_linear.txt
python run_loop_tool_sweep.py  2 1 0 | tee results/s2_v1_linear.txt

python run_loop_tool_sweep.py  4 4 0 | tee results/s4_v4_linear.txt
python run_loop_tool_sweep.py  4 1 0 | tee results/s4_v1_linear.txt

python run_loop_tool_sweep.py  8 4 0 | tee results/s8_v4_linear.txt
python run_loop_tool_sweep.py  8 1 0 | tee results/s8_v1_linear.txt
