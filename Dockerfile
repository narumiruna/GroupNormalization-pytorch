FROM nvidia/cuda:9.1-cudnn7-devel

LABEL maintainer="Narumi"

ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip \
    && pip3 install -U \
    http://download.pytorch.org/whl/cu91/torch-0.3.1-cp35-cp35m-linux_x86_64.whl \
    torchvision \
    && rm -rf ~/.cache/pip