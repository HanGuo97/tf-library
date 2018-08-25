# https://gitlab.com/nvidia/samples/blob/master/cuda/ubuntu16.04/cuda-samples/Dockerfile
# FROM nvidia/cuda:9.0-base-ubuntu16.04
# https://ngc.nvidia.com/registry/nvidia-tensorflow
FROM nvcr.io/nvidia/tensorflow:18.07-py3

# https://github.com/openai/gym/blob/master/Dockerfile
WORKDIR /research/TF-RLLibs
RUN mkdir -p TFLibrary && touch TFLibrary/__init__.py
COPY ./REQUIREMENTS.txt .
COPY ./setup.py .
RUN pip install -e . && \
	pip install -r REQUIREMENTS.txt

# Finally, upload our actual code
COPY . /research/TF-RLLibs



CMD /bin/bash
