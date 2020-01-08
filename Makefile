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

db.sqlite:
	docker run --rm -e DB_DIR=/io -v$(IO_DIR):/io --entrypoint python3 $(IMAGE) /server/tests/gen_sqlite_db.py

.PHONY: run-api
run-api: db.sqlite
	docker run --rm -p 8080:8080 -v$(IO_DIR):/io $(IMAGE) --ui --metadata-db=sqlite:///io/db.sqlite --state-db=sqlite://
