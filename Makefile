# Package configuration
PROJECT = athenian-api
IMAGE ?= athenian/api:dev

CI_REPOSITORY ?= https://github.com/athenianco/ci
CI_BRANCH ?= master
CI_PATH ?= .ci
MAKEFILE := $(CI_PATH)/Makefile.main
$(MAKEFILE):
	git clone --quiet --depth 1 -b $(CI_BRANCH) $(CI_REPOSITORY) $(CI_PATH);
-include $(MAKEFILE)

IO_DIR ?= $(PWD)/server/tests
ENV_FILE ?= .env

$(ENV_FILE):
	echo 'SENTRY_PROJECT=' >> $(ENV_FILE)
	echo 'SENTRY_KEY=' >> $(ENV_FILE)
	echo >> $(ENV_FILE)
	echo 'AUTH0_DOMAIN=' >> $(ENV_FILE)
	echo 'AUTH0_AUDIENCE=' >> $(ENV_FILE)

db.sqlite: $(ENV_FILE)
	docker run --rm -e DB_DIR=/io -v$(IO_DIR):/io --env-file $(ENV_FILE) --entrypoint python3 $(IMAGE) /server/tests/gen_sqlite_db.py

.PHONY: run-api
run-api: $(ENV_FILE) db.sqlite
	docker run --rm -p 8080:8080 -v$(IO_DIR):/io --env-file $(ENV_FILE) $(IMAGE) --ui --metadata-db=sqlite:///io/db.sqlite --state-db=sqlite://
