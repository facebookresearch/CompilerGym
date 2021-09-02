# Update the hardocded IP endpoints for the backend API. Defaults to localhost.
#
# Usage:   bash ./set_api_ip.sh <ip-address>
set -euo pipefail

main() {
    local ip="$1"

    for file in $(grep 127.0.0.1 --files-with-matches -R frontends/compiler_gym/src frontends/compiler_gym/package.json); do
        sed -i 's/127.0.0.1/'"$ip"'/' "$file"
    done
}
main $@
