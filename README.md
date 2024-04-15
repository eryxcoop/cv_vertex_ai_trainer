# CV Vertex AI Trainer
CLI used to remotely train a Computer Vision model using the Google Vertex AI platform.

We are assuming:
- You have a Google Cloud Platform account
- You have a Google Cloud Platform project
- You have a Google Cloud Storage bucket
- You have a Label Studio project with images and annotations
- In order to retrieve your Label Studio token, refer to: https://api.labelstud.io/#tag/Users/operation/api_current-user_token_list
- Your images are stored in a Google Cloud Storage bucket

# Setup requirements

`pip install -r requirements.txt`

`gcloud auth login`

If you are getting a warning about "project quota" run `gcloud auth application-default login`

run `pip install .` to install the package (or `pip install -e .` to install in editable mode)

# Usage

Create a `.toml` file following the example on `config.toml.example`

`cv-vertex-ai-trainer --help` to see the available commands

run `cv-vertex-ai-trainer -c config.toml`. Add `--local` to run the training locally.