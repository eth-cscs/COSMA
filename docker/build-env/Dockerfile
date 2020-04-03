FROM nvidia/cuda:10.1-devel-ubuntu18.04

WORKDIR /root

ARG MKL_VERSION=2020.0-088
ARG MPICH_VERSION=3.1.4

ENV DEBIAN_FRONTEND noninteractive
ENV MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl
ENV FORCE_UNSAFE_CONFIGURE 1
ENV MPICH_VERSION ${MPICH_VERSION}
ENV MKL_VERSION ${MKL_VERSION}

# Install basic tools
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends apt-utils gcc g++ gfortran git make unzip \
    vim wget pkg-config python3-pip curl environment-modules tcl \
    apt-transport-https ca-certificates gnupg software-properties-common \
    libhdf5-dev libgsl-dev imagemagick && \
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

# Install MKL
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.repos.intel.com/mkl all main' && \
    apt-get install -y -qq --no-install-recommends intel-mkl-${MKL_VERSION} && \
    rm -rf /var/lib/apt/lists/* && \ 
    echo "/opt/intel/lib/intel64\n/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf
