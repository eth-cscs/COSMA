ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG ENVPATH

# copy source files of the pull request into container
COPY . /src

# build SIRIUS
RUN spack -e $ENVPATH install

# # show the spack's spec
RUN spack -e $ENVPATH find -lcdv

# we need a fixed name for the build directory
# here is a hacky workaround to link ./spack-build-{hash} to ./spack-build
RUN cd /src && ln -s $(spack -e $ENVPATH location -b cosma) spack-build
