FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY sources.list /etc/apt/sources.list

ENV TZ Asia/Shanghai
ENV PYTHONPATH=$PYTHONPATH:/workspace/YOLOX


RUN rm /etc/apt/sources.list.d/cuda.list && rm /var/lib/apt/lists/* -vf && \
    apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y git tzdata vim build-essential software-properties-common ffmpeg libsm6 libxext6 cmake
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.7 python3.7-dev python3-pip python3-distutils && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python
RUN python -m pip install pqi && pqi use aliyun && \
    python -m pip install -U pip && python -m pip install -r requirements.txt && \
    git clone https://github.com.cnpmjs.org/cocodataset/cocoapi.git --depth=1 && cd cocoapi/PythonAPI/ && python setup.py install
