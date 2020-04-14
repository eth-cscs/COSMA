ARG BUILD_ENV

FROM $BUILD_ENV as builder

ARG BLAS

# Build COSMA
COPY . /COSMA

RUN COMPILERVARS_ARCHITECTURE=intel64 /opt/intel/bin/compilervars.sh && \
    mkdir /COSMA/build && cd /COSMA/build && \
    CC=mpicc CXX=mpicxx cmake .. \
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
RUN /root/libtree/libtree -d /root/COSMA.bundle/ \
      /root/COSMA-build/usr/bin/test.cosma \
      /root/COSMA-build/usr/bin/test.mapper \
      /root/COSMA-build/usr/bin/test.multiply \
      /root/COSMA-build/usr/bin/test.multiply_using_layout \
      /root/COSMA-build/usr/bin/test.pdgemm \
      /root/COSMA-build/usr/bin/test.scalar_matmul \
      # MKL dlopen's some of their libs, so we have to explicitly copy them over
      ${MKL_LIB}/libmkl_avx.so \
      ${MKL_LIB}/libmkl_avx2.so \
      ${MKL_LIB}/libmkl_avx512_mic.so \
      ${MKL_LIB}/libmkl_avx512.so \
      ${MKL_LIB}/libmkl_core.so \
      ${MKL_LIB}/libmkl_def.so \
      ${MKL_LIB}/libmkl_intel_thread.so \
      ${MKL_LIB}/libmkl_mc.so \
      ${MKL_LIB}/libmkl_mc3.so \
      ${MKL_LIB}/libmkl_sequential.so \
      ${MKL_LIB}/libmkl_tbb_thread.so \
      ${MKL_LIB}/libmkl_vml_avx.so \
      ${MKL_LIB}/libmkl_vml_avx2.so \
      ${MKL_LIB}/libmkl_vml_avx512_mic.so \
      ${MKL_LIB}/libmkl_vml_avx512.so \
      ${MKL_LIB}/libmkl_vml_cmpt.so \
      ${MKL_LIB}/libmkl_vml_def.so \
      ${MKL_LIB}/libmkl_vml_mc.so \
      ${MKL_LIB}/libmkl_vml_mc3.so

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
