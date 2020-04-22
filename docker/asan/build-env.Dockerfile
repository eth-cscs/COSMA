FROM ubuntu:18.04

WORKDIR /root

ARG MPICH_VERSION=3.3.2

ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_UNSAFE_CONFIGURE 1
ENV MPICH_VERSION ${MPICH_VERSION}

# Install basic tools
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    software-properties-common \
    build-essential gfortran pkg-config \
    git tar wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN wget -qO- "https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install MPICH ABI compatible with Cray's lib on Piz Daint
RUN wget -q https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar -xzf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure && \
    make install -j$(nproc) && \
    rm -rf /root/mpich-${MPICH_VERSION}.tar.gz /root/mpich-${MPICH_VERSION}

# Install OpenBLAS
ARG OPENBLAS_VERSION=0.3.9
RUN wget -qO - https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.tar.gz -O openblas.tar.gz && \
    tar -xzf openblas.tar.gz && \
    cd OpenBLAS-${OPENBLAS_VERSION}/ && \
    make DEBUG=1 -j$(nproc) && \
    make install NO_STATIC=1 PREFIX=/usr/local/ && \
    rm -rf /root/openblas.tar.gz /root/OpenBLAS-${OPENBLAS_VERSION}/ && \
    ldconfig

ARG NETLIB_SCALAPACK_VERSION=2.1.0

RUN wget -qO - http://www.netlib.org/scalapack/scalapack-${NETLIB_SCALAPACK_VERSION}.tgz -O scalapack.tar.gz && \
    tar -xzf scalapack.tar.gz && \
    cd scalapack-${NETLIB_SCALAPACK_VERSION} && \
    mkdir build && \
    cd build && \
    CC=mpicc FC=mpif90 cmake .. \
      -DBUILD_STATIC_LIBS=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Debug && \
    make -j$(nproc) && \
    make install && \
    rm -rf /root/scalapack.tar.gz /root/scalapack-${NETLIB_SCALAPACK_VERSION} && \
    ldconfig

# Add deployment tooling
RUN wget -q https://github.com/haampie/libtree/releases/download/v1.1.2/libtree_x86_64.tar.gz && \
    tar -xzf libtree_x86_64.tar.gz && \
    rm libtree_x86_64.tar.gz
