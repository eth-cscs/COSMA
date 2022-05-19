FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

WORKDIR /root
SHELL ["/bin/bash", "-c"]

ARG MPICH_VERSION=4.0.1
ARG OPENBLAS_VERSION=0.3.20
ARG NETLIB_SCALAPACK_VERSION=2.2.0

ENV DEBIAN_FRONTEND noninteractive
ENV MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl
ENV FORCE_UNSAFE_CONFIGURE 1
ENV MPICH_VERSION ${MPICH_VERSION}
ENV MKL_VERSION ${MKL_VERSION}

# reduce the minimum local dimension to allow all mpi ranks to take part 
# in testing
ENV COSMA_MIN_LOCAL_DIMENSION=32

# Install basic tools
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends \
      software-properties-common \
      build-essential gfortran pkg-config \
      git tar wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN wget -qO- "https://cmake.org/files/v3.22/cmake-3.22.1-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install MPICH ABI compatible with Cray's lib on Piz Daint
RUN wget -q https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar -xzf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure && \
    make install -j$(nproc) && \
    rm -rf /root/mpich-${MPICH_VERSION}.tar.gz /root/mpich-${MPICH_VERSION}

# Install OpenBLAS
RUN wget -qO - https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.tar.gz -O openblas.tar.gz && \
    tar -xzf openblas.tar.gz && \
    cd OpenBLAS-${OPENBLAS_VERSION}/ && \
    make TARGET=HASWELL NO_STATIC=1 -j$(nproc) && \
    make install TARGET=HASWELL NO_STATIC=1 PREFIX=/usr/local/ && \
    rm -rf /root/openblas.tar.gz /root/OpenBLAS-${OPENBLAS_VERSION}/ && \
    ldconfig

RUN wget -qO - http://www.netlib.org/scalapack/scalapack-${NETLIB_SCALAPACK_VERSION}.tgz -O scalapack.tar.gz && \
    tar -xzf scalapack.tar.gz && \
    cd scalapack-${NETLIB_SCALAPACK_VERSION} && \
    mkdir build && \
    cd build && \
    CC=mpicc FC=mpif90 cmake .. \
      -DBUILD_STATIC_LIBS=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    rm -rf /root/scalapack.tar.gz /root/scalapack-${NETLIB_SCALAPACK_VERSION} && \
    ldconfig

# Add deployment tooling
RUN mkdir -p /opt/libtree && \
    curl -Lfso /opt/libtree/libtree https://github.com/haampie/libtree/releases/download/v3.0.3/libtree_x86_64 && \
    chmod +x /opt/libtree/libtree

