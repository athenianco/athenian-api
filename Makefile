MAKEFLAGS += --no-builtin-rules

# Package configuration
PROJECT = athenian-api
SERVICE_NAME = api
IMAGE ?= gcr.io/athenian-1/api:latest

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

ifeq ($(DATABASE),postgres)
	export COMPOSE_PROJECT_NAME := test-athenian-api
	export POSTGRES_HOST_PORT := 5433
	export MEMCACHED_HOST_PORT := 11212
	export POSTGRES_USER := api
	export POSTGRES_PASSWORD := api
	export OVERRIDE_SDB := postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@0.0.0.0:$(POSTGRES_HOST_PORT)/state
	export OVERRIDE_MDB := postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@0.0.0.0:$(POSTGRES_HOST_PORT)/metadata
	export OVERRIDE_PDB := postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@0.0.0.0:$(POSTGRES_HOST_PORT)/precomputed
endif

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
	echo 'ATHENIAN_JIRA_INSTALLATION_URL_TEMPLATE=https://installation.athenian.co/jira/%s/atlassian-connect.json' >> $(ENV_FILE)
	echo 'ATHENIAN_DEV_ID=' >> $(ENV_FILE)

# Why do we touch?
# When you edit .env, it triggers rebuilding $(IO_DIR)/%.sqlite.
# mdb.sqlite is not rewritten if it already exists.
# Hence mdb.sqlite date is always older than .env.
# Hence $(IO_DIR)/%.sqlite is triggered every time.
# To avoid this, we touch the .sqlite file to bump its date after .env's.
$(IO_DIR)/%.sqlite: $(ENV_FILE)
	docker run --rm -e DB_DIR=/io -v$(IO_DIR):/io --env-file $(ENV_FILE) --entrypoint python3 $(IMAGE) /server/tests/gen_sqlite_db.py
	@touch $@

.PHONY: run-api
run-api: $(IO_DIR)/mdb.sqlite $(IO_DIR)/sdb.sqlite $(IO_DIR)/pdb.sqlite
	docker run $(DOCKER_RUN_EXTRA_ARGS) --rm -p 8080:8080 -v$(IO_DIR):/io --env-file $(ENV_FILE) $(IMAGE) --ui --no-google-kms --metadata-db=sqlite:///io/mdb.sqlite --state-db=sqlite:///io/sdb.sqlite --precomputed-db=sqlite:///io/pdb.sqlite

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

ifeq ($(DATABASE),postgres)
.PHONY: unittest-args
unittest-args:
	@echo "Environment variables for running tests:"
	@echo "- COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME}"
	@echo "- POSTGRES_HOST_PORT=$(POSTGRES_HOST_PORT)"
	@echo "- MEMCACHED_HOST_PORT=$(MEMCACHED_HOST_PORT)"
	@echo "- POSTGRES_USER=$(POSTGRES_USER)"
	@echo "- OVERRIDE_SDB=$(OVERRIDE_SDB)"
	@echo "- OVERRIDE_MDB=$(OVERRIDE_MDB)"
	@echo "- OVERRIDE_PDB=$(OVERRIDE_PDB)"

.PHONY: unittest-setup
unittest-setup:
	docker-compose -p $(COMPOSE_PROJECT_NAME) up -d postgres memcached
	sleep 5
	docker-compose exec postgres psql -c "create database state template 'template0' lc_collate 'C.UTF-8';" -U $(POSTGRES_USER)
	docker-compose exec postgres psql -c "create database metadata template 'template0' lc_collate 'C.UTF-8';" -U $(POSTGRES_USER)
	docker-compose exec postgres psql -c "create database precomputed template 'template0' lc_collate 'C.UTF-8';" -U $(POSTGRES_USER)

.PHONY: unittest-cleanup
unittest-cleanup:
	docker-compose down -v

.PHONY: unittest
unittest: unittest-args
	DATABASE=$(DATABASE) $(MAKE) unittest-setup
	-cd server && PYTHONPATH=. pytest $(VERBOSITY) $(TEST)
	DATABASE=$(DATABASE) $(MAKE) unittest-cleanup

.PHONY: unittest-no-setup
unittest-no-setup: unittest-args
	cd server && PYTHONPATH=. pytest $(VERBOSITY) $(TEST)
else
.PHONY: unittest
unittest:
	cd server && PYTHONPATH=. pytest $(VERBOSITY) $(TEST)
endif

upload-symbols.sh:
	git clone --depth 1 --branch main https://github.com/elastic/prodfiler-documentation /tmp/prodfiler-documentation
	cp /tmp/prodfiler-documentation/scripts/upload-symbols.sh .
	rm -rf /tmp/prodfiler-documentation
	chmod +x upload-symbols.sh

.PHONY: prodfiler-symbols
prodfiler-symbols: upload-symbols.sh
	for f in /usr/lib/x86_64-linux-gnu/libpython3.8.so.1.0 /usr/bin/python3.8 $$(ls -d /usr/lib/python3.8/lib-dynload/* | grep -v 38d); do \
	  dbg=$$(eu-unstrip -n -e $$f | cut -d" " -f4); \
	  if [ $$dbg != "-" ]; then \
	    ./upload-symbols.sh -u vadim@athenian.co -d $$f -g $$dbg; \
	  fi \
	done
