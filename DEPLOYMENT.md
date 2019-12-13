# How to deploy Athenian API

### Docker image

```
docker build -t athenian/api .
```

### Initialization

No special initialization is required.

### Environment

The server requires:

- `SENTRY_KEY` and `SENTRY_PROJECT` environment variables.
- Accessible PostgreSQL endpoint.
- Exposing the configured HTTP port outside.

### Configuration

Please follow the CLI help:

```
docker run -it --rm athenian/api --help
```

No configuration files are required.

### State

The server is currently stateless. No internal data structures are persisted anywhere.