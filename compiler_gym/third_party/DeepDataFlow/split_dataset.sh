#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Split out a subset of the DeepDataFlow LLVM bitcodes.
#
# Usage:
#     split_dataset /path/to/DeepDataFlow /path/to/outdir prefix-pattern suffix-pattern
set -eu

main() {
  local dataset_root="$1"
  local outdir="$(realpath $2)"
  local prefix="$3"
  local suffix="$4"

  cd "$dataset_root/bc"
  echo "Processing $prefix files ..."
  mkdir -p "$outdir/$prefix"
  find -L . -type f -name "$prefix"'*'"$suffix" | cat -n | while read n f; do cp "$f" "$outdir/$prefix/$n.bc"; done
}
main $@
