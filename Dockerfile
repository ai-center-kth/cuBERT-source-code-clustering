# Use nvidia/cuda image as base
FROM nvidia/cuda:10.2-base
CMD nvidia-smi

# Set working directory
WORKDIR /home/root/src

# Install necessary dependencies
RUN apt update && apt install -y --no-install-recommends \
    tmux \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools

# Set python3 as the default version
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 -q install pip --upgrade

# Copy requirements.txt and install the python packages
COPY src/requirements.txt /home/root/src
RUN pip3 install -r /home/root/src/requirements.txt

# Copy code to working directory
COPY src /home/root/src

# Make directory for the pre-trained model
RUN mkdir /home/root/src/model/tf_weights