FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt update -y && apt install -y git python3 python3-pip libgl1 libglib2.0-0 curl
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy files to run app
COPY src/ src/
COPY remote_training/ remote_training/
COPY mostro.toml mostro.toml
COPY config_example.toml config_example.toml
COPY sa.json sa.json

RUN pip3 install label-studio-sdk

CMD ["python3", "src/cli.py", "-c", "mostro.toml", "--local"]
