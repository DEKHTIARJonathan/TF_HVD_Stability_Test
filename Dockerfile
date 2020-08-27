# FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu16.04
FROM nvidia/cuda:11.0-runtime-ubuntu18.04

# Install GCC and other dependencies.
RUN apt-get update && \
    apt-get install --assume-yes \
        build-essential \
        git \
        wget \
        python3.7 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    gcc --version && \
    g++ --version

# Install tf-nightly and verify version.
RUN python3.7 -m pip install --upgrade pip && \
    pip3.7 install --no-cache --no-cache-dir tf-nightly && \
    python3.7 -c "import tensorflow as tf; print(tf.__version__)" && \
    pip3.7 install --no-cache --no-cache-dir pytest

WORKDIR /tmp/openmpi_source

# Download and install open-mpi.
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz && \
    tar xvf openmpi-4.0.4.tar.gz && \
    cd openmpi-4.0.4 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install

# Set the path for the binary, libs and headers to be discoverable
ENV LD_LIBRARY_PATH=/usr/local/lib/openmpi
RUN ldconfig

ENV HOROVOD_GPU_OPERATIONS=NCCL
ENV HOROVOD_WITH_TENSORFLOW=1
ENV HOROVOD_WITHOUT_PYTORCH=1
ENV HOROVOD_WITHOUT_MXNET=1
RUN pip3.7 install --no-cache --no-cache-dir git+https://github.com/horovod/horovod.git

WORKDIR /workspace
# Install tests.
RUN git clone https://github.com/DEKHTIARJonathan/TF_HVD_Stability_Test.git /workspace
RUN pip3.7 install -r requirements.txt --user
