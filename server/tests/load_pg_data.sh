#!/bin/bash
# This bash script is used for local development along with the docker-compose.yml
# file to ease the access to real data.

# Environment variables needed:
# - POSTGRES_SOURCE_HOST: host of PG to dump the data from
# - POSTGRES_SOURCE_PORT: port of PG to dump the data from
# - POSTGRES_SOURCE_USER: user of PG to dump the data from
# - POSTGRES_SOURCE_PASSWORD: password of PG to dump the data from
# - POSTGRES_USER: user of local PG to restore the data to

# Options:
# - --drop-db drop databases if existing before restore
# - --reuse-existing-dump reuse the dump files if existing

# Pass db names ass positional arguments to restore only some databases
set -e

export PGPASSWORD=${POSTGRES_SOURCE_PASSWORD}
DUMPS_DIR=/db_dumps

dump() {
    echo "Dumping database $1..."
    pg_dump --host ${POSTGRES_SOURCE_HOST} --port=${POSTGRES_SOURCE_PORT} --no-owner --no-privileges --username=${POSTGRES_SOURCE_USER} --dbname=$1 | grep -v -E 'CREATE (EXTENSION|COMMENT).*pg_repack' > ${DUMPS_DIR}/$1.dump
    echo "Dump done."
}

restore() {
    echo "Restoring database $1..."
    createdb -U ${POSTGRES_USER} --lc-collate='C.UTF-8' -T template0 $1
    file=${DUMPS_DIR}/$1.dump
    psql -U ${POSTGRES_USER} -d $1 -f $file
    echo "Restore done."
}


drop_db=0
reuse_existing_dump=0
while [ 1 ]; do
    if [ "$1" = "--drop-db" ]; then
        drop_db=1
        shift
    elif [ "$1" = "--reuse-existing-dump" ]; then
        reuse_existing_dump=1
        shift
    else
        break
    fi
done
    
databases=${@:-"state precomputed persistentdata metadata"}

mkdir -p ${DUMPS_DIR}

for db in $databases; do
    if [ ! -f "${DUMPS_DIR}/${db}.dump" ] || [ "$reuse_existing_dump" = 0 ]; then
        dump ${db}
    fi

    if [ $drop_db -eq 1 ]; then
        dropdb -U ${POSTGRES_USER} "$db" || true
    fi

    restore ${db}
done
