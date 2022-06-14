#!/bin/sh

set -eu

echo "Running isort..."
isort --check .

echo "Running black (chorny)"
chorny --check .

echo "Running flake8"
flake8
