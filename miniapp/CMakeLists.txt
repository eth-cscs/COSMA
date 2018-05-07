cmake_minimum_required(VERSION 3.5)

project(tests)

set(BIN_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TEST_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

####################
#  Lib Dependancy  #
####################

include_directories("${PRJ_ROOT_DIR}/src")

################
#  Build test  #
################

add_lib_executable(carma-statistics
  carma)

if (CARMA_WITH_CUDA)
    cuda_add_lib_executable(carma-miniapp
      carma
      carmacu
      ${carma_deps})

    cuda_add_lib_executable(temp-miniapp
      carma
      carmacu
      ${carma_deps})
else()
    add_lib_executable(carma-miniapp
      carma
      ${carma_deps})

    add_lib_executable(temp-miniapp
      carma
      ${carma_deps})
endif()
