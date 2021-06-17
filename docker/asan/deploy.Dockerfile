ARG BUILD_ENV

FROM $BUILD_ENV as builder

ARG BLAS

# Build COSMA
COPY . /COSMA

# reduce the minimum local dimension to allow all mpi ranks to take part 
# in testing
ENV COSMA_MIN_LOCAL_DIMENSION=32

RUN mkdir /COSMA/build && cd /COSMA/build && \
    CC=mpicc CXX=mpicxx cmake .. \
      -DCOSMA_WITH_TESTS=ON \
      -DCOSMA_BLAS=OPENBLAS \
      -DCOSMA_SCALAPACK=CUSTOM \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS_DEBUG="-g -Og -fno-omit-frame-pointer -fsanitize=address,undefined" \
      -DCMAKE_INSTALL_PREFIX=/root/COSMA-build && \
      make -j$(nproc) && \
      make install && \
      rm -rf /COSMA

RUN /root/libtree/libtree \
      --chrpath \
      -d /root/COSMA.bundle/ \
      /root/COSMA-build/bin/test.cosma \
      /root/COSMA-build/bin/test.mapper \
      /root/COSMA-build/bin/test.multiply \
      /root/COSMA-build/bin/test.multiply_using_layout \
      /root/COSMA-build/bin/test.pdgemm \
      /root/COSMA-build/bin/test.scalar_matmul

FROM ubuntu:18.04

COPY --from=builder /root/COSMA.bundle /root/COSMA.bundle

# Make it easy to call our binaries.
ENV PATH="/root/COSMA.bundle/usr/bin:$PATH"

RUN echo "/root/COSMA.bundle/usr/lib/" > /etc/ld.so.conf.d/cosma.conf && ldconfig

WORKDIR /root/COSMA.bundle/usr/bin

# I'm not getting ASAN_OPTIONS=suppressions=file to work, so just disable leak detection for now.
ENV ASAN_OPTIONS=detect_leaks=false


