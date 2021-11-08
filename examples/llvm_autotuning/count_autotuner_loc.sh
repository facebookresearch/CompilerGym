# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Run this script without arguments from any directory:
#
#     bash llvm_autotuning/count_autotuner_loc.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

for file in $(ls "$SCRIPT_DIR"/autotuners/*.py | grep -v _test | grep -v __init__ | sort); do
    echo -n $(basename "$file")
    sloccount "$file" | grep 'Total Physical Source Lines of Code' | cut -d'=' -f2
done
