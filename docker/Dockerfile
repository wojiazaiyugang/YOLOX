FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY sources.list /etc/apt/sources.list
COPY TensorRT-7.1.3.4 /workspace/TensorRT-7.1.3.4

ENV TZ Asia/Shanghai
ENV PYTHONPATH=$PYTHONPATH:/workspace/YOLOX
ENV LD_LIBRARY_PATH=/workspace/TensorRT-7.1.3.4/lib:$LD_LIBRARY_PATH:/workspace/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/lib


RUN rm /etc/apt/sources.list.d/cuda.list && rm /var/lib/apt/lists/* -vf && \
    apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install git tzdata vim build-essential software-properties-common ffmpeg libsm6 libxext6 cmake -y && \
    add-apt-repository -r -y ppa:deadsnakes/ppa && \
    apt install -y python3.7 python3.7-dev python3-pip python3-distutils && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python && \
    python -m pip install pqi && pqi use aliyun && \
    python -m pip install -U pip cython && python -m pip install scikit-build onnx onnx-simplifier && python -m pip install -r requirements.txt && \
    git clone https://github.com.cnpmjs.org/cocodataset/cocoapi.git --depth=1 && cd cocoapi/PythonAPI/ && python setup.py install && \
    cd /workspace && git clone https://github.com.cnpmjs.org/NVIDIA-AI-IOT/torch2trt --depth=1 && cd torch2trt && python setup.py install && \
    cd /workspace/TensorRT-7.1.3.4/python && python -m pip install tensorrt-7.1.3.4-cp37-none-linux_x86_64.whl
