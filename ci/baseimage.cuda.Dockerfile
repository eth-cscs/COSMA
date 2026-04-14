FROM ubuntu:24.04 as builder

ARG CUDA_ARCH=90

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

ENV PATH="/spack/bin:${PATH}"

ENV MPICH_VERSION=4.3.2

RUN apt-get -y update

RUN apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y --no-install-recommends gcc g++ gfortran clang libomp-14-dev git make unzip file \
  vim wget pkg-config python3-pip python3-dev cython3 python3-pythran curl tcl m4 cpio automake meson \
  xz-utils patch patchelf apt-transport-https ca-certificates gnupg software-properties-common perl tar bzip2 \
  liblzma-dev libbz2-dev

# get latest version of spack
RUN git clone -b releases/v1.1 https://github.com/spack/spack.git

# set the location of packages built by spack
RUN spack config add config:install_tree:root:/opt/local
# set cuda_arch for all packages
RUN spack config add packages:all:variants:cuda_arch=${CUDA_ARCH}

# add local repo for cosma and tiled-mm
COPY ./spack_repo /spack_repo
RUN spack repo add /spack_repo/cosma

# find all external packages
RUN spack external find --all --exclude python --exclude meson

# find compilers
RUN spack compiler find

# install yq (utility to manipulate the yaml files)
RUN wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_arm64 && chmod a+x /usr/local/bin/yq

# install MPICH
RUN spack install mpich@${MPICH_VERSION} %gcc

# for the MPI hook
RUN echo $(spack find --format='{prefix.lib}' mpich) > /etc/ld.so.conf.d/mpich.conf
RUN ldconfig

# create environments for several configurations and install dependencies
RUN mkdir -p /cosma-env-cuda/costa /cosma-env-cuda/tiled-mm
RUN spack env create -d /cosma-env-cuda && \
    spack -e /cosma-env-cuda add "cosma@=master +cuda +tests +scalapack +shared %gcc  ^mpich" && \
    spack -e /cosma-env-cuda add "tiled-mm@=master" && \
    spack -e /cosma-env-cuda develop -p "./tiled-mm tiled-mm" && \
    spack -e /cosma-env-cuda add "cuda@12" && \
    spack -e /cosma-env-cuda add "costa@=master" && \
    spack -e /cosma-env-cuda develop -p "./costa costa" && \
    spack -e /cosma-env-cuda develop -p /src cosma@master && \
    spack -e /cosma-env-cuda install --only=dependencies --fail-fast

# RUN spack env create -d /cosma-env-cuda-gpu-direct && \
#     spack -e /cosma-env-cuda-gpu-direct add "cosma@master +cuda +tests +scalapack +shared +gpu_direct %gcc  ^mpich " && \
#     spack -e /cosma-env-cuda-gpu-direct add "tiled-mm@master" && \
#     spack -e /cosma-env-cuda-gpu-direct add "costa@master" && \
#     spack -e /cosma-env-cuda-gpu-direct add "cuda@12" && \
#     spack -e /cosma-env-cuda-gpu-direct develop -p /src cosma@master && \
#     spack -e /cosma-env-cuda-gpu-direct install --only=dependencies --fail-fast

# RUN spack env create -d /cosma-env-cuda-nccl && \
#     spack -e /cosma-env-cuda-nccl add "cosma@master +cuda +tests +scalapack +shared +nccl  %gcc ^mpich " && \
#     spack -e /cosma-env-cuda-nccl add "tiled-mm@2.3.1" && \
#     spack -e /cosma-env-cuda-nccl add "costa@master" && \
#     spack -e /cosma-env-cuda-nccl add "cuda@12" && \
#     spack -e /cosma-env-cuda-nccl develop -p /src cosma@master && \
#     spack -e /cosma-env-cuda-nccl install --only=dependencies --fail-fast

# RUN spack env create -d /cosma-env-cpu && \
#     spack -e /cosma-env-cpu add "cosma@master ~cuda +tests +scalapack +shared %gcc  ^mpich " && \
#     spack -e /cosma-env-cpu add "costa@master" && \
#     spack -e /cosma-env-cpu develop -p /src cosma@master && \
#     spack -e /cosma-env-cpu install --only=dependencies --fail-fast
