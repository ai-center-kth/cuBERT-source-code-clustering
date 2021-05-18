# Use nvidia/cuda image as base
# Tag this image as "scc-base"
FROM nvidia/cuda:10.2-base
CMD nvidia-smi

# Install necessary dependencies
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools

# Upgrade pip
RUN pip3 -q install pip --upgrade

# Use the base image by its name and install python packages
RUN pip3 install torch \
    transformers \
    tensorflow-gpu \
    tensor2tensor \
    nltk \
    numpy \
    pandas \
    sklearn \
    matplotlib \
    seaborn \
    tqdm \
    astunparse \
    python_minifier \
    yapf \
    gsutil

# Set python3 as default
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /home/scc/src

# Copy code to working directory
COPY src /home/scc/src

# Make directory for the pre-trained model
RUN mkdir /home/scc/src/model/tf_weights