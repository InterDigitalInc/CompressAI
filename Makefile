PYTORCH_DOCKER_IMAGE = pytorch/pytorch:1.6.0-cuda10.1-cudnn7
PYTHON_DOCKER_IMAGE = python:3.8-buster

GIT_DESCRIBE = $(shell git describe --first-parent)
ARCHIVE = compressai.tar.gz

.PHONY: help
help:
	@echo  'Docker targets:'
	@echo  '  docker          - based on latest pytorch image (with GPU support)'
	@echo  '  docker-cpu      - based on latest python3 image (smaller image without GPU support)'


.PHONY: docker
docker:
	@git archive --format=tar.gz HEAD > docker/${ARCHIVE}
	@cd docker && \
		docker build \
		--build-arg PYTORCH_IMAGE=${PYTORCH_DOCKER_IMAGE} \
		--build-arg WITH_JUPYTER=0 \
		--progress=auto \
		-t compressai:${GIT_DESCRIBE} .
	@rm docker/${ARCHIVE}


.PHONY: docker-cpu
docker-cpu:
	@git archive --format=tar.gz HEAD > docker/${ARCHIVE}
	@cd docker && \
		docker build \
		-f Dockerfile.cpu \
		--build-arg BASE_IMAGE=${PYTHON_DOCKER_IMAGE} \
		--build-arg WITH_JUPYTER=0 \
		--progress=auto \
		-t compressai:${GIT_DESCRIBE}-cpu .
	@rm docker/${ARCHIVE}
