#!/bin/bash
# This bash script is used for local development along with the docker-compose.yml
# file to ease the access to real data.

# Environment variables needed:
# - POSTGRES_SOURCE_HOST: host of PG to dump the data from
# - POSTGRES_SOURCE_PORT: port of PG to dump the data from
# - POSTGRES_SOURCE_USER: user of PG to dump the data from
# - POSTGRES_SOURCE_PASSWORD: password of PG to dump the data from
# - POSTGRES_USER: user of local PG to restore the data to
set -xe

export PGPASSWORD=${POSTGRES_SOURCE_PASSWORD}

dump() {
    echo "Dumping database $1..."
    pg_dump --host ${POSTGRES_SOURCE_HOST} --port=${POSTGRES_SOURCE_PORT} --no-owner --username=${POSTGRES_SOURCE_USER} -Fc --dbname=$1 > /db_dumps/$1.dump
    echo "Dump done."
}

restore() {
    echo "Restoring database $1..."
    createdb -U ${POSTGRES_USER} --lc-collate='C.UTF-8' -T template0 $1
    pg_restore -x -Fc -O -U ${POSTGRES_USER} -d $1 /db_dumps/$1.dump
    echo "Restore done."
}

mkdir -p db_dumps
for db in state precomputed
do
    dump ${db}
    restore ${db}
done
rm -rf db_dumps
