# Learning and Evaluating Source Code Embeddings using cuBERT

## Getting Started

We provide a docker image for easy deployment of the container in order to get up and running quickly. However, if you would like to skip this step then you can head over directly to the src-folder and follow the instructions [here](/src).

### Docker
#### Prerequisites

- [Docker](https://www.docker.com/)
- [NVIDIA-Drivers](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)


Build the docker image
```
docker build -t scc .
```

Start the container, exposing the GPUs and open a terminal prompt.
```
docker run -it -p 8888:8888 --gpus all scc /bin/bash
```

The container is now up and running with all the necessary dependencies installed.
Instructions for running the code are available [here](/src)

