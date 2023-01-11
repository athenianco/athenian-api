#!/bin/bash

set -u
err=0
trap '(( err |= $? ))' ERR

echo "Running isort..."
isort --check .

echo "Running flake8"
flake8

# chorny is not capable alone to enforce and check our coding formatting style
# so add-trailing-comma must be called also; a clean GIT tree is required to
# check that nothing is changed after calling those commands
echo "Running add-trailing-comma + black(chorny)"
if ! [ -z "$(git diff HEAD)" ]; then
    echo "Cannot check code formatting with unclean GIT working tree, sorry."
    exit 1
fi

find . -path './athenian/api/sentry_native' -prune -o \( -name '*.py' -print \) |
    xargs add-trailing-comma --py36-plus

(chorny .)

if ! [ -z "$(git diff HEAD)" ]; then
    echo "Some files modified after code formatting check."
    git status --porcelain
    err=1
fi

exit $err
