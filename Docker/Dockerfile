FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install wget -y
RUN apt-get install python3 -y
RUN apt-get install python3-dev -y
RUN apt-get install python3-pip -y
RUN pip install pybind11
RUN pip install open3d
RUN pip install open3d-cpu
RUN pip install numpy


WORKDIR /workspace

#Установка open3d
RUN wget https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz \
    && mkdir -p /lib/open3d \
    && tar -xf open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz -C /lib/open3d --strip-components=1


RUN apt-get install libc++-dev libc++abi-dev -y

RUN apt install -y build-essential cmake libopenblas-dev libblas-dev libomp-dev -y

#Установка LibTorch(PyTorch) для С++ CPU version
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
RUN apt install unzip -y
RUN unzip *.zip

RUN rm *.tar.xz
RUN rm *.zip

##Установка onnx runtime для С++ CPU version
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz \
    && mkdir onnxruntime \
    && tar -xf onnxruntime-linux-x64-1.22.0.tgz  -C /workspace/onnxruntime --strip-components=1
    
RUN rm *.tgz 

#Расскоментить если перестала рабоать графика тогда к sudo нужно будет добавить -E при cmake и make open3d пакетов
#RUN useradd -m -s /bin/bash newuser && \
#    echo 'newuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER newuser
