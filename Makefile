PYTHON ?= python

.PHONY: test lint compile
test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests

compile:
	$(PYTHON) -m compileall -q src
