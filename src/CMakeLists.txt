cmake_minimum_required(VERSION 3.5)

###############
#  Src files  #
###############

set(sources
    context.cpp
    buffer.cpp
    communicator.cpp
    hybrid_communicator.hpp
    local_multiply.cpp
    multiply.cpp
    matrix.cpp
    layout.cpp
    mapper.cpp
    statistics.hpp
    strategy.cpp
    interval.cpp
    timer.hpp
    math_utils.cpp)

###################
#  Cosma Library  #
###################
message(${cosma_deps})
add_library(cosma ${sources})
#target_link_libraries(cosma ${cosma_deps} /opt/cray/dmapp/7.1.1-6.0.7.0_34.3__g5a674e0.ari/lib64/libdmapp.so)
target_link_libraries(cosma ${cosma_deps})
# target_include_directories(cosma PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src)
