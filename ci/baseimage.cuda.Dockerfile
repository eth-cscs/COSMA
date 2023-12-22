FROM ubuntu:22.04 as builder

ARG CUDA_ARCH=60

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

ENV PATH="/spack/bin:${PATH}"

ENV MPICH_VERSION=3.4.3

ENV CMAKE_VERSION=3.27.9

RUN apt-get -y update

RUN apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y --no-install-recommends gcc g++ gfortran clang libomp-14-dev git make unzip file \
  vim wget pkg-config python3-pip python3-dev cython3 python3-pythran curl tcl m4 cpio automake meson \
  xz-utils patch patchelf apt-transport-https ca-certificates gnupg software-properties-common perl tar bzip2 \
  liblzma-dev libbz2-dev

# install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar zxvf cmake.tar.gz --strip-components=1 -C /usr

# get latest version of spack
RUN git clone -b v0.21.0 https://github.com/spack/spack.git

# set the location of packages built by spack
RUN spack config add config:install_tree:root:/opt/local
# set cuda_arch for all packages
RUN spack config add packages:all:variants:cuda_arch=${CUDA_ARCH}

# find all external packages
RUN spack external find --all

# find compilers
RUN spack compiler find

# install yq (utility to manipulate the yaml files)
RUN wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_386 && chmod a+x /usr/local/bin/yq

# change the fortran compilers: for gcc the gfortran is already properly set and the change has no effect; add it for clang
RUN yq -i '.compilers[0].compiler.paths.f77 = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml && \
    yq -i '.compilers[0].compiler.paths.fc = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml  && \
    yq -i '.compilers[1].compiler.paths.f77 = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml && \
    yq -i '.compilers[1].compiler.paths.fc = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml

# install MPICH
RUN spack install mpich@${MPICH_VERSION} %gcc

# for the MPI hook
RUN echo $(spack find --format='{prefix.lib}' mpich) > /etc/ld.so.conf.d/mpich.conf
RUN ldconfig

# create environments for several configurations and install dependencies
RUN spack env create -d /cosma-env-cuda && \
    spack -e /cosma-env-cuda add "cosma@master %gcc +cuda +tests +scalapack +shared ^mpich " && \
    spack -e /cosma-env-cuda develop -p /src cosma@master && \
    spack -e /cosma-env-cuda install --only=dependencies --fail-fast

RUN spack env create -d /cosma-env-cuda-gpu-direct && \
    spack -e /cosma-env-cuda-gpu-direct add "cosma@master %gcc +cuda +tests +scalapack +shared +gpu_direct ^mpich " && \
    spack -e /cosma-env-cuda-gpu-direct develop -p /src cosma@master && \
    spack -e /cosma-env-cuda-gpu-direct install --only=dependencies --fail-fast

RUN spack env create -d /cosma-env-cpu && \
    spack -e /cosma-env-cpu add "cosma@master %gcc ~cuda +tests +scalapack +shared ^mpich " && \
    spack -e /cosma-env-cpu develop -p /src cosma@master && \
    spack -e /cosma-env-cpu install --only=dependencies --fail-fast
