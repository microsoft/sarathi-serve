.PHONY: help lint lint/flake8 lint/black lint/isort format format/black format/autopep8 format/isort
.DEFAULT_GOAL := help

lint/flake8: ## check style with flake8
	flake8 sarathi

lint/black: ## check style with black
	black --check sarathi

lint/isort: ## check style with isort
	isort --check-only --profile black sarathi

lint: lint/black lint/isort ## check style

format/black: ## format code with black
	black sarathi

format/autopep8: ## format code with autopep8
	autopep8 --in-place --recursive sarathi/

format/isort: ## format code with isort
	isort --profile black sarathi

format: format/isort format/black ## format code
