FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY sources.list /etc/apt/sources.list

ENV TZ Asia/Shanghai


RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install git tzdata vim build-essential software-properties-common ffmpeg libsm6 libxext6 cmake -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.7 python3.7-dev python3-pip python3-distutils && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python && \
    python -m pip install pqi && pqi use aliyun && \
    python -m pip install -U pip cython && python -m pip install scikit-build && python -m pip install -r requirements.txt && \
    git clone https://github.com.cnpmjs.org/cocodataset/cocoapi.git --depth=1 && cd cocoapi/PythonAPI/ && python setup.py install && \
    cd /workspace && git clone https://github.com.cnpmjs.org/NVIDIA-AI-IOT/torch2trt --depth=1 && cd torch2trt && python setup.py install
