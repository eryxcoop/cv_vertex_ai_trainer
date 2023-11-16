from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("remote_training/train_requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="cv-vertex-ai-trainer",
    description="CLI used to remotely train a Computer Vision model using the Google Vertex AI platform.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eryxcoop/cv-vertex-ai-trainer",
    author="Eryx",
    version="0.0.1",
    packages=find_packages(),
    package_data={"cv-vertex-ai-trainer": ["config_example.toml"]},
    entry_points={"console_scripts": ["cv-vertex-ai-trainer=src.cli:main"]},
    install_requires=requirements,
    python_requires=">=3.10",
)
