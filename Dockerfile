FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04


# set environment variable
ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_DEBUGGER_SOFTWARE_PREEMPTION 1
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics


# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl docker.io vim wget ca-certificates


# Install python and pip
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip


# Install pytorch
RUN pip3 install torch torchvision torchaudio

# install other stuff to prevent ImportError: libGL.so.1: 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install dos2unix


# set up folders
WORKDIR /app
COPY . /app

# Install other stuff
RUN pip install -r requirements.txt

RUN dos2unix run.sh

# autostart
#ENTRYPOINT ["python", "train.py"]
#ENTRYPOINT ["bash", "run.sh"]