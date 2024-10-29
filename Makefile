.PHONY: pypi-clean pypi-deploy pypi-local-build local.run local.clean

LOCAL_CONFIG_FILE := local.toml # Set here the name of your local config file
MOSTRO_CONFIG_FILE := mostro.toml
DOCKER_IMAGE_NAME := cv-vertex-ai-trainer-mostro
PYTHONPATH := $(shell pwd)

pypi-clean:
	rm -rf build/ dist/ *.egg-info/

pypi-deploy: pypi-install-deps
	python setup.py sdist bdist_wheel
	twine upload dist/*
	make pypi-clean

pypi-local-build:
	pip install .

pypi-install-deps:
	source venv/bin/activate
	pip install twine

local.run: local.clean
	PYTHONPATH=${PYTHONPATH} python src/cli.py -c ${LOCAL_CONFIG_FILE} --local

local.clean:
	rm -rf dataset/
	rm -f annotations.zip
	rm -f yolov8n.pt

mostro.docker.build:
	docker build . -t ${DOCKER_IMAGE_NAME}

mostro.docker.run:
	@if [ -z "$(container_name)" ]; then \
		echo "Error: CONTAINER_NAME parameter is missing. Please run this command like 'make mostro.docker.run container_name=CONTAINER_NAME'"; \
		exit 1; \
	else \
		echo "Running with container_name: $(container_name)"; \
		docker run -e PYTHONPATH=. --device nvidia.com/gpu=all --ipc=host --name ${container_name} ${DOCKER_IMAGE_NAME}; \
	fi

mostro.docker.copy_dataset_and_results:
	@if [ -z "$(container_name)" ]; then \
		echo "Error: CONTAINER_NAME parameter is missing. Please run this command like 'make mostro.docker.copy_dataset_and_results container_name=CONTAINER_NAME'"; \
		exit 1; \
	else \
		echo "Running with container_name: $(container_name)"; \
		docker cp ${container_name}:/app/dataset/ .; \
    fi