# How to deploy Athenian API

### Docker image

```
docker build -t athenian/api .
```

### Initialization

```
docker run -it --rm --entrypoint python3 athenian/api -m athenian.api.models.state postgres://user:password@host:port/state
docker run -it --rm --entrypoint python3 athenian/api -m athenian.api.models.precomputed postgres://user:password@host:port/precomputed
docker run -it --rm --entrypoint python3 athenian/api -m athenian.api.models.persistentdata postgres://user:password@host:port/persistentdata
```

### Environment

The server requires:

- (optional) `SENTRY_KEY`, `SENTRY_PROJECT` and `SENTRY_ENV` environment variables to enable error logging.
- (optional) `ATHENIAN_MAX_CLIENT_SIZE` to limit the maximum request body size (256KB by default).
- (optional) `ATHENIAN_MAX_LOAD`to set the maximum load in abstract units the server accepts before rejecting requests with HTTP 503; the default value is 12.
- (optional) `ATHENIAN_SEGMENT_KEY` to enable user action tracking in [Segment](https://segment.com/).
- (optional) `GOOGLE_ANALYTICS` tag to track Swagger UI.
- `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET` environment variables to enable authorization.
- `ATHENIAN_DEFAULT_USER` environment variable that points to the Auth0 user ID used for unauthorized public requests.
- `ATHENIAN_INVITATION_KEY` environment variable with the passphrase for encrypting invitation URLs.
- `ATHENIAN_INVITATION_URL_PREFIX` environment variable which specifies the invitation URL beginning.
- `ATHENIAN_JIRA_INSTALLATION_URL_TEMPLATE` environment variable which specifies the JIRA integration installation link; "%s" will be replaced with the unique account key.
- `GOOGLE_KMS_PROJECT`, `GOOGLE_KMS_KEYRING`, `GOOGLE_KMS_KEYNAME` environment variables to specify the [Google Key Management Service](https://cloud.google.com/kms/docs) symmetric encrypt/decrypt key.
- `GOOGLE_KMS_SERVICE_ACCOUNT_JSON` (file path) or `GOOGLE_KMS_SERVICE_ACCOUNT_JSON_INLINE` (JSON) to access Google KMS.
- Accessible PostgreSQL endpoint with the metadata.
- Accessible PostgreSQL endpoint with the server state.
- Accessible PostgreSQL endpoint with the precomputed objects.
- Exposing the configured HTTP port outside.

### Configuration

Please follow the CLI help:

```
docker run -it --rm athenian/api --help
```

No configuration files are required.

`--memcached` may be specified to cache user profiles, computed auxiliaries, etc. It is optional.

### State

The server's state such as user settings, etc., is stored in a SQL database specified with `--state-db`.
Besides, the state implicitly depends on the cache (`--memcached`) and the precomputed objects (`--precomputed-db`).
So running a sparkling clean and fresh API server requires:

- Wiping the state DB, the precomputed objects DB, and the persistentdata DB.
- Re-launching memcached.