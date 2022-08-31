#!/bin/sh

set -eu

echo $PGUSER > /dev/null
echo $PGPASSWORD > /dev/null
dbname=check_sdb_migrations

export ATHENIAN_INVITATION_KEY=777 # needed by migrations

psql -c "DROP DATABASE IF EXISTS $dbname" -h 0.0.0.0 -p 5432  -d postgres
psql -c "CREATE DATABASE $dbname TEMPLATE 'template0' LC_COLLATE 'C.UTF-8'" -h 0.0.0.0 -p 5432

tmpfile=
cleanup() {
    # restore alembic.ini
    if [ -n "$tmpfile" ]; then
        mv "$tmpfile" alembic.ini
    fi
}
# backup alembic.ini
if [ -f alembic.ini ]; then
    tmpfile=$(mktemp)
    cp -a alembic.ini $tmpfile
    trap cleanup EXIT
fi

python3 -m athenian.api.models.state "postgresql://$PGUSER:$PGPASSWORD@0.0.0.0:5432/$dbname"

alembic-autogen-check

psql -c "DROP DATABASE $dbname" -h 0.0.0.0 -p 5432  -d postgres
