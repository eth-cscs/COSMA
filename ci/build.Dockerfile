ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG ENVPATH

# copy source files of the pull request into container
COPY . /src

# # show the spack's spec
RUN spack -e $ENVPATH find -lcdv

# build COSTA and Tiled-MM with current @master branch
RUN git clone -b master https://github.com/eth-cscs/COSTA.git $ENVPATH/costa
RUN git clone -b master https://github.com/eth-cscs/Tiled-MM.git $ENVPATH/tiled-mm

# build packages
RUN spack -e $ENVPATH install

# we need a fixed name for the build directory
# here is a hacky workaround to link ./spack-build-{hash} to ./spack-build
RUN cd /src && ln -s $(spack -e $ENVPATH location -b cosma) spack-build
