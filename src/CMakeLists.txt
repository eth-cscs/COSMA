cmake_minimum_required(VERSION 3.5)

##################
#  Generic Part  #
##################

#add_definitions(-DDEBUG)

###############
#  Src files  #
###############

set(sources
library.cpp
communication.cpp
rectmm.cpp
matrix.cpp
layout.cpp
mapper.cpp
)

####################
#  Lib Dependancy  #
####################

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###################
#  Carma Library  #
###################

add_library(carma ${sources}
${blas_deps}
${mpi_deps}
)

