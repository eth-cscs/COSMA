ARG BUILD_ENV

FROM $BUILD_ENV as builder

ARG BLAS

# Add deployment tooling
RUN wget -q https://github.com/haampie/bundler/releases/download/v0.1.3/bundler_x86_64.tar.gz && \
    tar -xzf bundler_x86_64.tar.gz && \
    rm bundler_x86_64.tar.gz

# Build COSMA
COPY . /COSMA

RUN COMPILERVARS_ARCHITECTURE=intel64 /opt/intel/bin/compilervars.sh intel64 && \
    mkdir /COSMA/build && cd /COSMA/build && \
    CC=mpicc CXX=mpicxx FC=mpif90 FCCPP=cpp cmake .. \
      -DCOSMA_WITH_TESTS=ON \
      -DCUDA_PATH=/usr/local/cuda \
      -DCOSMA_BLAS=${BLAS} \
      -DCOSMA_SCALAPACK=MKL \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr && \
      make -j$(nproc) && \
      make DESTDIR=/root/COSMA-build install && \
      rm -rf /COSMA

ENV MKL_LIB=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64

# Run linuxdeploy, and add a bunch of libs that are dlopen'ed by mkl
RUN /root/bundler/bundler -d /root/COSMA.bundle/ \
      -e /root/COSMA-build/usr/bin/test.cosma \
      -e /root/COSMA-build/usr/bin/test.pdgemm \
      -e /root/COSMA-build/usr/bin/test.multiply \
      -e /root/COSMA-build/usr/bin/test.mapper \
      -e /root/COSMA-build/usr/bin/test.scalar_matmul \
      -e /root/COSMA-build/usr/bin/test.multiply_using_layout \
      # MKL dlopen's some of their libs, so we have to explicitly copy them over
      -l ${MKL_LIB}/libmkl_tbb_thread.so \
      -l ${MKL_LIB}/libmkl_intel_thread.so \
      -l ${MKL_LIB}/libmkl_sequential.so \
      -l ${MKL_LIB}/libmkl_def.so \
      -l ${MKL_LIB}/libmkl_vml_def.so \
      -l ${MKL_LIB}/libmkl_vml_cmpt.so \
      -l ${MKL_LIB}/libmkl_mc.so \
      -l ${MKL_LIB}/libmkl_vml_mc.so \
      -l ${MKL_LIB}/libmkl_mc3.so \
      -l ${MKL_LIB}/libmkl_vml_mc3.so \
      -l ${MKL_LIB}/libmkl_avx.so \
      -l ${MKL_LIB}/libmkl_vml_avx.so \
      -l ${MKL_LIB}/libmkl_avx2.so \
      -l ${MKL_LIB}/libmkl_vml_avx2.so \
      -l ${MKL_LIB}/libmkl_avx512_mic.so \
      -l ${MKL_LIB}/libmkl_vml_avx512_mic.so \
      -l ${MKL_LIB}/libmkl_avx512.so \
      -l ${MKL_LIB}/libmkl_vml_avx512.so \
      -l ${MKL_LIB}/libmkl_core.so 

FROM ubuntu:18.04

# This is the only thing necessary really from nvidia/cuda's ubuntu18.04 runtime image
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

COPY --from=builder /root/COSMA.bundle /root/COSMA.bundle

# Make it easy to call our binaries.
ENV PATH="/root/COSMA.bundle/usr/bin:$PATH"

RUN echo "/root/COSMA.bundle/usr/lib/" > /etc/ld.so.conf.d/cosma.conf && ldconfig

WORKDIR /root/COSMA.bundle/usr/bin
