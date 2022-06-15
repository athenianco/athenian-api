#!/bin/sh

set -eu

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

(find . -name '*.py' | xargs add-trailing-comma --py36-plus)

(chorny .)

if ! [ -z "$(git diff HEAD)" ]; then
    echo "Some files modified after code formatting check."
    git status --porcelain
    exit 1
fi
