ARG BUILD_ENV_IMAGE

ARG BUILD_ENV_TAG

FROM $BUILD_ENV_IMAGE:$BUILD_ENV_TAG as builder

# Build COSMA
COPY . /COSMA

# The deployment tool creates an appdir, and an appdir requires an icon and desktop file, which is a bit redundant of course, but let's just go with it
RUN printf "[Desktop Entry] \n\
Name=COSMA \n\
Exec=test.cosma %%F \n\
Icon=COSMA \n\
Type=Application \n\
Categories=Science; \n\
" > /root/COSMA.desktop

RUN convert -size 16x16 xc:white /root/COSMA.png

# Add deployment tooling
RUN wget -q https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage

# Working with appimages requires fuse, which inside docker requires it to be mounted and --cap-add SYS_ADMIN
# so instead just extract the appimage.
RUN chmod +x linuxdeploy-x86_64.AppImage && ./linuxdeploy-x86_64.AppImage --appimage-extract

RUN mkdir /COSMA/build && cd /COSMA/build && \
    CC=mpicc CXX=mpicxx FC=mpif90 FCCPP=cpp cmake .. \
      -DCOSMA_WITH_TESTS=ON \
      -DCUDA_PATH=/usr/local/cuda \
      -DCOSMA_BLAS=CUDA \
      -DCOSMA_SCALAPACK=MKL \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr && \
      make -j$(nproc) && \
      make DESTDIR=/root/COSMA.AppDir install

ENV MKL_LIB=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64

# Run linuxdeploy, and add a bunch of libs that are dlopen'ed by mkl
RUN /root/squashfs-root/AppRun --appdir /root/COSMA.AppDir/ \
        -e /root/COSMA.AppDir/usr/bin/test.cosma \
        -i /root/COSMA.png \
        -d /root/COSMA.desktop \
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

COPY --from=builder /root/COSMA.AppDir /root/COSMA.AppDir

# Make it easy to call our binaries.
ENV PATH="/root/COSMA.AppDir/usr/bin:$PATH"

WORKDIR /root/COSMA.AppDir/usr/bin
