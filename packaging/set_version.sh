#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# ------------------------------------------------------------------------------
# This script is used to update any source files in the repo that have a version
# string embedded in them. To set a new verison, call this script with the new
# version number and todays date in yyyy-mm-dd format. For example:
#
#     bash packaging/set_version.sh 0.1.12 2021-09-20
set -euo pipefail

main() {
    local version="$1"
    local date="$2"

    if [[ ! -f VERSION ]]; then
        echo "fatal: VERSION file not found" >&2
        exit 1
    fi
    echo "$version" > VERSION
    echo "Wrote VERSION"

    sed -e 's/^version:.*$/version: '"$version"'/' -i CITATION.cff
    set -x
    sed -e 's/^date-released:.*$/date-released: '"$date"'/' -i CITATION.cff
    echo "Wrote CITATION.cff"

    sed -e 's/^compiler_gym==.*$/compiler_gym=='"$version"'/' -i www/requirements.txt
    echo "Wrote www/requirements.txt"

    git add -p VERSION CITATION.cff www/requirements.txt
}
main $@
