#!/bin/sh

set -eu

projdir="$(dirname $(dirname $0))"
cd "$projdir"

docker run --rm -v ${PWD}:/local openapitools/openapi-generator-cli generate -i /local/docs/openapi.yaml -g python-aiohttp -o /local/server_new --git-repo-id athenianco/athenian-api --package-name athenian.api
black server_new
