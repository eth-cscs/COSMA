FROM ubuntu:20.04

WORKDIR /root
SHELL ["/bin/bash", "-c"]

ARG MKL_VERSION=2020.4-912
ARG MPICH_VERSION=4.0.1

ENV DEBIAN_FRONTEND noninteractive
ENV MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl
ENV FORCE_UNSAFE_CONFIGURE 1
ENV MPICH_VERSION ${MPICH_VERSION}
ENV MKL_VERSION ${MKL_VERSION}

# reduce the minimum local dimension to allow all mpi ranks to take part 
# in testing
ENV COSMA_MIN_LOCAL_DIMENSION=32

# Install basic tools
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git tar wget curl gpg-agent && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN wget -qO- "https://cmake.org/files/v3.22/cmake-3.22.1-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install MPICH ABI compatible with Cray's lib on Piz Daint
RUN wget -q https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar -xzf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure --disable-fortran && \
    make install -j$(nproc) && \
    rm -rf /root/mpich-${MPICH_VERSION}.tar.gz /root/mpich-${MPICH_VERSION}

# Install MKL
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.repos.intel.com/mkl all main' && \
    apt-get install -y -qq --no-install-recommends intel-mkl-64bit-${MKL_VERSION} && \
    rm -rf /var/lib/apt/lists/* && \
    echo "/opt/intel/lib/intel64\n/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
    ldconfig

# Add deployment tooling
RUN mkdir -p /opt/libtree && \
    curl -Lfso /opt/libtree/libtree https://github.com/haampie/libtree/releases/download/v3.0.3/libtree_x86_64 && \
    chmod +x /opt/libtree/libtree

