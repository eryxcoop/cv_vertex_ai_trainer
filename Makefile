.PHONY: pypi-clean pypi-deploy pypi-local-build local.run local.clean

LOCAL_CONFIG_FILE := local.toml # Set here the name of your local config file
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
	rm -f annotations.json
	rm -f yolov8n.pt