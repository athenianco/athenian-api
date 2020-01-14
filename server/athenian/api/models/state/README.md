# Alembic migrations

Generic single-database configuration.

You need to have `alembic.ini`. The easiest way to generate it is to run

```
python3 -m athenian.api.models.state postgres://user:password@host:port/database
```

Replace `postgres://user:password@host:port/database` with the actual DB connection string in
[SQLAlchemy format](https://docs.sqlalchemy.org/en/13/core/engines.html).

It is possible to define `OFFLINE` environment variable (to any non-empty value) so that
the SQL operations dump is written to the standard output while the actual DB stays untouched.
You still need to specify the DB connection string though.

### Upgrading DB to the most recent schema version

```
alembic upgrade head
```

### Generating a new migration

```
alembic revision -m "Description of the migration"
```

Then edit the newly generated file by hand. Note: you need to have
[`black` installed](https://github.com/psf/black).