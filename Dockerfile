# See: https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated
#FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Install GCC, Python3.8 and other dependencies.
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/America/Monterrey /etc/localtime && \
    DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes \
        build-essential \
        git \
        wget \
        cmake \
        curl \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        python3.8 \
        python3.8-dev \
        python3-pip \
        python3.8-distutils && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    gcc --version && \
    g++ --version

# Install tf-nightly and verify version.
RUN python3.8 -m pip install --upgrade pip && \
    pip3.8 install --no-cache --no-cache-dir tf-nightly && \
    python3.8 -c "import tensorflow as tf; print(tf.__version__)"

WORKDIR /tmp/openmpi_source

# Download and install open-mpi.
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz && \
    tar xvf openmpi-4.0.4.tar.gz && \
    cd openmpi-4.0.4 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install

# Set the path for OpenMPI binaries, libs and headers to be discoverable
ENV LD_LIBRARY_PATH=/usr/local/lib/openmpi
RUN ldconfig

ENV HOROVOD_GPU_OPERATIONS=NCCL
ENV HOROVOD_WITH_TENSORFLOW=1
ENV HOROVOD_WITHOUT_PYTORCH=1
ENV HOROVOD_WITHOUT_MXNET=1

RUN pip3.8 install --no-cache --no-cache-dir \
        git+https://github.com/horovod/horovod.git

WORKDIR /workspace

COPY requirements.txt /tmp/
RUN  pip3.8 install --no-cache --no-cache-dir -r /tmp/requirements.txt
