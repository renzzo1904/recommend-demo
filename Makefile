
TEST_DIR := test

# Default target
all: install test lint format

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	@echo "Running tests..."
	python -m pytest -vv test/test_*.py

lint:
	@echo "Linting code..."
    	pylint --disable=R,C **/*.py

format:
	@echo "Formatting code..."
	black **/*.py







