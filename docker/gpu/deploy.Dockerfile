ARG BUILD_ENV

FROM $BUILD_ENV as builder

ARG BLAS

# Build COSMA
COPY . /COSMA

RUN mkdir /COSMA/build && cd /COSMA/build && \
    CC=mpicc CXX=mpicxx cmake .. \
      -DCOSMA_WITH_TESTS=ON \
      -DCUDA_PATH=/usr/local/cuda \
      -DCOSMA_BLAS=CUDA \
      -DCOSMA_SCALAPACK=CUSTOM \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/root/COSMA-build && \
      make -j$(nproc) && \
      make install && \
      rm -rf /COSMA

# Run linuxdeploy, and add a bunch of libs that are dlopen'ed by mkl
RUN /root/libtree/libtree \
      -d /root/COSMA.bundle/ \
      --chrpath \
      --strip \
      /root/COSMA-build/bin/test.cosma \
      /root/COSMA-build/bin/test.mapper \
      /root/COSMA-build/bin/test.multiply \
      /root/COSMA-build/bin/test.multiply_using_layout \
      /root/COSMA-build/bin/test.pdgemm \
      /root/COSMA-build/bin/test.scalar_matmul

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
