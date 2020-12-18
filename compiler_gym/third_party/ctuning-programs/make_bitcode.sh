#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

indent() {
  sed 's/^/    /'
}

make_bitcode() {
  local clang="$(pwd)/$1"
  local llvm_link="$(pwd)/$2"
  local outpath="$3"
  local dir="$4"

  # NOTE(cummins): The LLVM release does not include a full set of standard
  # includes. Hack arond this on macOS.
  if [[ -d /Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include ]]; then
    clang="$clang -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include/machine"
  fi

  echo "Building $outpath ..."
  rm -f "$dir/*.bc"
  TARGET=a.bc CC="$clang" LD="$llvm_link" CFLAGS="-emit-llvm -c" OBJ_EXT=bc LD_FLAGS= make -C "$dir" 2>&1 | indent
  cp -v "$dir/a.bc" "$outpath" 2>&1 | indent
  echo
}
make_bitcode $@
