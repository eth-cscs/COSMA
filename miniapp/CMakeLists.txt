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

add_lib_executable(cosma-miniapp
  cosma
  ${cosma_deps})

add_lib_executable(temp-miniapp
  cosma
  ${cosma_deps})

add_lib_executable(cosma-statistics
  cosma
  ${cosma_deps})

add_lib_executable(test-strategy
  cosma
  ${cosma_deps})
