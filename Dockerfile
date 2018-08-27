# https://gitlab.com/nvidia/samples/blob/master/cuda/ubuntu16.04/cuda-samples/Dockerfile
# FROM nvidia/cuda:9.0-base-ubuntu16.04
# https://ngc.nvidia.com/registry/nvidia-tensorflow
FROM nvcr.io/nvidia/tensorflow:18.07-py3

# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
RUN apt-get update && apt-get install -y \
    parallel \
 && rm -rf /var/lib/apt/lists/*

# https://github.com/openai/gym/blob/master/Dockerfile
WORKDIR /research/TF-RLLibs
RUN mkdir -p TFLibrary && touch TFLibrary/__init__.py
COPY ./REQUIREMENTS.txt .
COPY ./README.md .
COPY ./setup.py .
RUN pip install -e . && \
	pip install -r REQUIREMENTS.txt

# Finally, upload our actual code
COPY . /research/TF-RLLibs



CMD /bin/bash