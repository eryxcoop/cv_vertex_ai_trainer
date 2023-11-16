.PHONY: pypi-clean pypi-deploy pypi-local-build

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
