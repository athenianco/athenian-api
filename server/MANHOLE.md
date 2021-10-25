# Manhole

API backdoor to execute arbitrary code in the requests.

### Where the code executes

See `async def manhole()` in [athenian/api/\_\_init\_\_.py](athenian/api/connexion.py).

### Writing the code

See [tests/test_manhole.py](tests/test_manhole.py) for the examples.

- You can freely use `await`.
- `await handler(request)` produces the original response.
- Assign to `response` to override the response. If the value is not `None`, the regular handler
will not be executed.
- Assign to `athenian.api.trace_sample_rate_manhole` to change the traces sampling rate in Sentry.
- The user is resolved inside the `handler`, so it is not possible to check them before calling
`await handler(request)`. Thus `request.uid` emerges after the call.

### Installing the code

Forward `memcached` to `localhost:7001`:

```
gcloud container clusters get-credentials production-cluster --zone us-east1-c --project athenian-1
kubectl port-forward service/mcrouter-mcrouter 7001:5000 -n mcrouter
```

`pip3 install aiomcache` and execute the following in IPython (that's important: IPython, not
the regular REPL):

```
from aiomcache import Client
client = Client("0.0.0.0", 7001)
print((await client.get(b"manhole", b"")).decode())
payload = """...your code..."""
await client.set(b"manhole", payload.encode())
```
