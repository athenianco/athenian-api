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

DOCKER_RUN_EXTRA_ARGS ?= -it

IO_DIR ?= $(PWD)/server/tests
ENV_FILE ?= .env

$(ENV_FILE):
	echo 'SENTRY_PROJECT=' >> $(ENV_FILE)
	echo 'SENTRY_KEY=' >> $(ENV_FILE)
	echo >> $(ENV_FILE)
	echo 'AUTH0_DOMAIN=' >> $(ENV_FILE)
	echo 'AUTH0_AUDIENCE=' >> $(ENV_FILE)
	echo 'AUTH0_CLIENT_ID=' >> $(ENV_FILE)
	echo 'AUTH0_CLIENT_SECRET=' >> $(ENV_FILE)
	echo >> $(ENV_FILE)
	echo 'ATHENIAN_DEFAULT_USER=github|60340680' >> $(ENV_FILE)
	echo 'ATHENIAN_INVITATION_KEY=we-are-the-best' >> $(ENV_FILE)
	echo 'ATHENIAN_INVITATION_URL_PREFIX=http://localhost:3000/i/' >> $(ENV_FILE)

$(IO_DIR)/%.sqlite: $(ENV_FILE)
	docker run --rm -e DB_DIR=/io -v$(IO_DIR):/io --env-file $(ENV_FILE) --entrypoint python3 $(IMAGE) /server/tests/gen_sqlite_db.py

.PHONY: run-api
run-api: $(IO_DIR)/mdb.sqlite $(IO_DIR)/sdb.sqlite
	docker run $(DOCKER_RUN_EXTRA_ARGS) --rm -p 8080:8080 -v$(IO_DIR):/io --env-file $(ENV_FILE) $(IMAGE) --ui --metadata-db=sqlite:///io/mdb.sqlite --state-db=sqlite:///io/sdb.sqlite

.PHONY: invitation-link
invitation-link: $(IO_DIR)/sdb.sqlite
	docker run --rm -e DB_DIR=/io -v$(IO_DIR):/io --env-file $(ENV_FILE) --entrypoint python3 $(IMAGE) -m athenian.api.invite_admin sqlite:///io/sdb.sqlite

.PHONY: gkwillie
gkwillie: $(IO_DIR)/sdb.sqlite
	docker run --rm -e DB_DIR=/io -v$(IO_DIR):/io --env-file $(ENV_FILE) --entrypoint python3 $(IMAGE) -m athenian.api.create_default_user sqlite:///io/sdb.sqlite

.PHONY: clean
clean: fixtures-clean

.PHONY: fixtures-clean
fixtures-clean:
	rm -rf $(IO_DIR)/mdb.sqlite $(IO_DIR)/sdb.sqlite
