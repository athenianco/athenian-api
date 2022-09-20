#!/bin/bash -ex

apt-get update
apt-get install -y binutils curl
curl -sL https://sentry.io/get-cli/ | bash


for f in $(find "/server" -name '*.cpython-*-x86_64-linux-gnu.so'); do
  objcopy --only-keep-debug --compress-debug-sections=zlib $f $f.debug
  sentry-cli upload-dif --auth-token $SENTRY_AUTH_TOKEN --org $SENTRY_ORG --project $SENTRY_PROJECT --include-sources --type elf $f.debug
  rm $f.debug
done

sentry-cli upload-dif --auth-token $SENTRY_AUTH_TOKEN --org $SENTRY_ORG --project $SENTRY_PROJECT --include-sources --type elf /usr/lib/debug/.build-id