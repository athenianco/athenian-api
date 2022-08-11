# Guide to adding new API endpoints

## Edit [`athenian/api/openapi/openapi.yml`](athenian/api/openapi/openapi.yml)

The spec lives in a different repository and should be committed and released there.

Useful links:

* [Introduction.](https://swagger.io/docs/specification/basic-structure/)
* [Data types.](https://swagger.io/docs/specification/data-models/data-types/)

Things to remember:

* `openapi.yml` is a regular Jinja2 template. The parameters are specified in `self.add_api(arguments=...)` at `athenian/api/__init__.py`.
* `openapi.yml` is also a regular YAML! You can use pointers, multi-line strings (`|-`), etc.
* OpenAPI is based on [JSON Schema](https://json-schema.org/) but there are very few differences.
* We employ `tags` to sort endpoints in Swagger UI. `default` is required for simpler JavaScript client bindings and is stripped away on the server because it duplicates the endpoints in Swagger UI.
* The Python handler will be searched at `"x-openapi-router-controller"."operationId"`.

Gotchas:

* Python server [does not support `oneOf` pointing at other `components/schemas`](https://github.com/spec-first/connexion/issues/691).
* JavaScript client does not support `oneOf` and `anyOf`. We are using a workaround with `allOf` and Jinja2 conditionals.
* Forgetting `security:\n- bearerAuth: []` will lead to obscure error messages at server startup.
* Forgetting `additionalProperties: false` in `components/schemas` allows the frontend to send any weird shit that the server silently discards.

## Generate the new models with [`../docs/generate_server.sh`](../docs/generate_server.sh)

This script gives you `server_new` with the generated web models and the processed spec.

Notes:

* script must be run from repository root directory
* `openapi-generator` doesn't seem to support x-openapi-router-controller right now, controller paths
  should be checked manually


## Check the differences between the specs

You run `meld server/spec/openapi.yml server_new/spec/openapi.yml` and inspect the changes.
You port all the changes back to the original spec, **but only for the added endpoints**.

## Copy the new models

You copy the new model files from `server_new/athenian/api/models/` to `server/athenian/api/models/web/`.

## Run `black` over the new models

You run `black` over the added model files to reduce the amount of manual work.

## Edit the new models

You have to manually edit the new files:

1. Remove the headers.
2. Remove `from_dict()` method. I have no clue why it generates because it is already present in the parent class.
3. Fix the `base_model_` location, and remove many unused imports.
4. Add `typing.Optional` to the arguments of `__init__`. They are indeed optional there even though are declared as `required` in the spec.
5. Remove the typed docstrings of the generated properties in favor of `typing`.
6. Some docstrings contain double empty new lines which do not pass `flake8`. `black` is not perfect :(
7. Copy the class desription from the spec over those lame "DO NOT EDIT" warnings.
8. When in doubt, look at the many examples around.

Finally, run `python3 -m athenian.api.models.web` to regenerate `server/athenian/api/models/web/__init__.py`.

## Add the implementation

Change the type of the request to `athenian.api.request.AthenianWebRequest` to enjoy better IDE hints.
You've got `request.*db` (mdb, sdb, pdb) to access the databases. `request.cache` is an optional
(can be `None`) `aiomcache.Client`.

Tips:

* Use `@cached` decorator for easy caching.
* Open `async with request.*db.connection():` if you are going to make several queries to the same DB.
* Prefer idempotent behavior (`PUT` over `PATCH`) wherever possible to simplify your life and avoid complex problems in the future.
* Return `http.HTTPStatus.NOT_IMPLEMENTED` (501) from unimplemented endpoints.
