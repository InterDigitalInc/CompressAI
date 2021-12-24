.DEFAULT_GOAL := help

PYTORCH_DOCKER_IMAGE = pytorch/pytorch:1.8.1-cuda11.1-cudnn8
PYTHON_DOCKER_IMAGE = python:3.8-buster

GIT_DESCRIBE = $(shell git describe --first-parent)
ARCHIVE = compressai.tar.gz

src_dirs := compressai tests examples docs

.PHONY: help
help: ## Show this message
	@echo "Usage: make COMMAND\n\nCommands:"
	@grep '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' | cat


# Check style and linting
.PHONY: check-black check-isort check-flake8 check-mypy static-analysis

check-black: ## Run black checks
	@echo "--> Running black checks"
	@black --check --verbose --diff $(src_dirs)

check-isort: ## Run isort checks
	@echo "--> Running isort checks"
	@isort --check-only $(src_dirs)

check-flake8: ## Run flake8 checks
	@echo "--> Running flake8 checks"
	@flake8 $(src_dirs)

check-mypy: ## Run mypy checks
	@echo "--> Running mypy checks"
	@mypy

static-analysis: check-black check-isort check-flake8 check-mypy ## Run all static checks


# Apply styling
.PHONY: style

style: ## Apply style formating
	@echo "--> Running black"
	@black $(src_dirs)
	@echo "--> Running isort"
	@isort $(src_dirs)


# Run tests
.PHONY: tests coverage

tests:  ## Run tests
	@echo "--> Running Python tests"
	@pytest -x -m "not slow" --cov compressai --cov-append --cov-report= ./tests/

coverage: ## Run coverage
	@echo "--> Running Python coverage"
	@coverage report
	@coverage html


# Build docs
.PHONY: docs

docs: ## Build docs
	@echo "--> Building docs"
	@cd docs && SPHINXOPTS="-W" make html


# Docker images
.PHONY: docker docker-cpu
docker: ## Build docker image
	@git archive --format=tar.gz HEAD > docker/${ARCHIVE}
	@cd docker && \
		docker build \
		--build-arg PYTORCH_IMAGE=${PYTORCH_DOCKER_IMAGE} \
		--build-arg WITH_JUPYTER=0 \
		--progress=auto \
		-t compressai:${GIT_DESCRIBE} .
	@rm docker/${ARCHIVE}

docker-cpu: ## Build docker image (cpu only)
	@git archive --format=tar.gz HEAD > docker/${ARCHIVE}
	@cd docker && \
		docker build \
		-f Dockerfile.cpu \
		--build-arg BASE_IMAGE=${PYTHON_DOCKER_IMAGE} \
		--build-arg WITH_JUPYTER=0 \
		--progress=auto \
		-t compressai:${GIT_DESCRIBE}-cpu .
	@rm docker/${ARCHIVE}
