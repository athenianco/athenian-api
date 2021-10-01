# Security of the API

## Users

Each user is a member of one or more accounts.
The user can be either an admin or a non-admin (regular). Those roles differ in permissions.
Each account has at least one admin user.

### github.com

API user always corresponds to a GitHub user.
Hence the user always has a full name and a nickname (login).
The API accesses the user's email address if it exists in their profile.

### Enterprise

The integration with Okta mirrors the external userbase through OpenID.

## Authentication

API supports three authentication mechanisms.

1. No authentication. The requests do not carry any special headers. The effective user sets to
`ATHENIAN_DEFAULT_USER`, which is [*@gkwillie*](https://github.com/gkwillie) in all our environments.
*@gkwillie* is a regular member of account #1 - Athenian dogfood - and exists for anonymous access
to the API for demonstration purposes.
2. [JWT (JSON Web Token)](https://jwt.io/). This is the main authentication mode that all the users
of the [Athenian application](https://app.athenian.co) follow. We delegate the user database and
the client-side handshakes to [Auth0](https://auth0.com). The API checks the incoming JWTs according
to the preloaded Auth0 public key and reads the user details from Auth0 server.
3. [API Key](https://swagger.io/docs/specification/authentication/api-keys/). The user can issue
one or more API Keys that function similar to [GitHub Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).
API delegates the verification and private key storage to [Google Cloud Key Management (KMS)](https://cloud.google.com/security-key-management).
The tokens are not saved anywhere and cannot be recovered.

## Modelled threats

### Broken authentication

A hacker can break the existing authentication and forge the user.

We delegate the work with JWTs and API Keys to reliable partners - Auth0 and Google. Both mechanisms
exclude brute force attacks. The code that parses JWTs is covered with an extensive test suite.

### Anonymous users access unauthorized accounts

A user can make an API request, such as fetch information of a specific client or change their
settings without authentication.

We allow anonymous access as *@gkwillie* but it is constrained to access Athenian dogfood account only.

### Authenticated users access unauthorized accounts

Endpoints in the `/filter` group return information about PRs, releases, JIRA issues in the user-supplied
repositories.

We check whether the requesting user has access to each provided repository on all the endpoints.
To avoid logic bugs in SQL when a buggy query returns data of unrelated accounts, we pin
each and every row in every table in all our databases to the owner account and filter by it.

`/become` endpoint allows the members of Athenian organization except *@gkwillie* to forge the identity
and act on behalf on any other account user. The list of users that can access `/become` is hardcoded.

### Confidential data disclosure

An anonymous or authorized user accesses the personal information of other users in external accounts.

`/account{id}/details` lists the members of the calling user's accounts. We check whether the caller
may access the specified account ID.
We don't store the personal information such as user full names or emails in any database that
the API server can directly access. The API works with personal information exclusively through Auth0.

### API leaks source code of the clients

The hacker intrudes in the API server and is able to download the source code of the clients.

The API does not fetch the source code. It only stores Git metadata.

### API leaks infrastructure secrets

The API returns an error message that contains secret tokens, logins and passwords, internal URLs, etc.

The debug mode is disabled in production and the server never responds with a stack trace or error messages
with secrets.
We delete all the environment variables in the API server after it starts.
We don't expose secrets as environment variables in our Continuous Integration with GitHub Actions.
GitHub notifies us if we push a commit with hardcoded infrastructure secrets.

### SQL injections

API has to process e.g. user-provided repository names that may contain SQL injection code.

We don't insert user-provided text in plaintext SQL queries. The SQL composition bases on
[SQLAlchemy](https://docs.sqlalchemy.org/) that performs all the required text sanitization and escaping.
Request data validation executes automatically following the OpenAPI specification in
[zalando/connexion](https://github.com/zalando/connexion).

### Man in the middle

Somebody who may intercept the HTTP traffic between the user and the API is able to read the responses.

All interaction with the API happens through HTTPS. API talks to Auth0 and Google KMS through HTTPS.

### HTTP protocol and other low-level attacks

For example, [request smuggling](https://portswigger.net/research/http2#h2desync) through HTTP protocol
downgrade.

API server bases on [`aiohttp`](https://docs.aiohttp.org/en/stable/) to serve HTTP.
API server operates HTTP 1.1 that further upgrades to HTTP 2 by the [Google Load Balancer](https://cloud.google.com/load-balancing).
We monitor the vulnerabilities both in `aiohttp` and Google LB and upgrade the packages as soon as
possible. We employ automated vulnerability scanning by [semgrep](https://semgrep.dev/),
[Snyk](https://snyk.io/), [GitHub CodeQL](https://codeql.github.com/), as well as
receive automated package upgrades from [dependabot](https://dependabot.com/).

### Arbitrary code execution

A malicious API request leads to arbitrary code execution on the server.

We have no `exec` and `eval` calls in the API except the ["manhole"](server/MANHOLE.md).
The way Python works excludes the buffer overflow attacks.
The API endpoints follow the strict schema and don't serve a custom query language or similar.

### Denial of Service

The user sends a request that leads to huge API workload and either makes the server unresponsive
or restarts it.

We've seen three scenarios:

1. Bad API request loads the DB and leads to global API DoS for all the users.
2. Bad API request leads to 100% CPU on the processing server instance. It either times out other
requests or returns HTTP 503 Service Unavailable.
3. Bad API request leads to extraordinary memory consumption and the server instance restarts.

Measures taken:

1. Request size limit.
2. Request processing timeout.
3. Request complexity limitations on eligible endpoints.
4. Request rate limits on all the endpoints.
5. Monitoring and alerts.

### Missing an attack

When somebody hacks the API we never realize and let the attack continue.

We report API errors to [Sentry](https://sentry.io/),
log to [Google Cloud Logging](https://cloud.google.com/logging),
as well as monitor the server metrics in Prometheus and built-in Google Cloud
inspection. We have set up alerts on each critical metric and monitor them.

### Improper code management

We forget to update a legacy endpoint after a refactoring and it exposes data without proper
authentication or security access checks.

The API is driven by the [OpenAPI specification](https://github.com/athenian/api-spec), and
the coded endpoints satisfy it, instead of the specification being updated after code changes.
That makes the mentioned situation nearly impossible.

### Deploying hacked code

We deploy a server that was previously hacked.

Checking access to API source code relies on GitHub. It is technically impossible to deploy a new
API version without a review approval by somebody else in the engineering team. We maintain
the built Docker images in [Google Container Registry](https://cloud.google.com/container-registry).

### Security audits

We forget to model an important threat and the server is vulnerable to another attack.

Athenian regularly undergoes external security audits and penetration tests.

## SLA

We audit new vulnerabilities every week and fix them within 2 working days.
