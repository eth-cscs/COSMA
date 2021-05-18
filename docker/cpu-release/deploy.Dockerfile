ARG BUILD_ENV

FROM $BUILD_ENV as builder

# Build COSMA
COPY . /COSMA

RUN COMPILERVARS_ARCHITECTURE=intel64 /opt/intel/bin/compilervars.sh && \
    mkdir /COSMA/build && cd /COSMA/build && \
    CC=mpicc CXX=mpicxx cmake .. \
      -DCOSMA_WITH_TESTS=ON \
      -DCOSMA_BLAS=MKL \
      -DCOSMA_SCALAPACK=MKL \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/root/COSMA-build && \
      make -j$(nproc) && \
      make install && \
      rm -rf /COSMA

ENV MKL_LIB=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64

# Run linuxdeploy, and add a bunch of libs that are dlopen'ed by mkl
RUN /root/libtree/libtree --chrpath --strip -d /root/COSMA.bundle/ \
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

COPY --from=builder /root/COSMA.bundle /root/COSMA.bundle

# Make it easy to call our binaries.
ENV PATH="/root/COSMA.bundle/usr/bin:$PATH"

RUN echo "/root/COSMA.bundle/usr/lib/" > /etc/ld.so.conf.d/cosma.conf && ldconfig

WORKDIR /root/COSMA.bundle/usr/bin
