#!/bin/sh

set -eu

projdir="$(dirname $(dirname $0))"
cd "$projdir"

uid=$(id -u)
gid=$(id -g)

docker run --rm \
       -u $uid:$gid \
       -v ${PWD}:/local \
       openapitools/openapi-generator-cli \
       generate \
       -i /local/docs/openapi.yaml \
       -g python-aiohttp \
       -o /local/server_new \
       --git-repo-id athenianco/athenian-api \
       --package-name athenian.api
black server_new
