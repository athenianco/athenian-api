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
export DUMPS_DIR=/db_dumps

dump() {
    echo "Dumping database $1..."
    pg_dump --host ${POSTGRES_SOURCE_HOST} --port=${POSTGRES_SOURCE_PORT} --no-owner --username=${POSTGRES_SOURCE_USER} --dbname=$1 | grep -v -E 'CREATE (EXTENSION|COMMENT).*pg_repack' > ${DUMPS_DIR}/$1.dump
    echo "Dump done."
}

restore() {
    echo "Restoring database $1..."
    createdb -U ${POSTGRES_USER} --lc-collate='C.UTF-8' -T template0 $1
    file=${DUMPS_DIR}/$1.dump
    mime_type=$(file -b -i $file | cut -d ' ' -f1 | rev | cut -c 2- | rev)
    if [ $mime_type == "text/plain" ]; then
        psql -U ${POSTGRES_USER} -d $1 -f $file
    else
        pg_restore -x -O -U ${POSTGRES_USER} -d $1 $file
    fi
    echo "Restore done."
}

mkdir -p ${DUMPS_DIR}
for db in state precomputed persistentdata metadata
do
    if [ ! -f "${DUMPS_DIR}/${db}.dump" ]; then
        dump ${db}
    fi

    restore ${db}
done
rm ${DUMPS_DIR}/*
