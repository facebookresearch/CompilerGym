# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Update the hardocded IP endpoints for the backend API. Defaults to localhost.
#
# Usage:   bash www/set_api_ip.sh <ip-address>
set -euo pipefail

main() {
    local ip="$1"

    for file in $(grep 127.0.0.1 --files-with-matches -R www/frontends/compiler_gym/src www/frontends/compiler_gym/package.json); do
        sed -i 's/127.0.0.1/'"$ip"'/' "$file"
    done
}
main $@
