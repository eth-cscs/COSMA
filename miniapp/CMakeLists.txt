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

add_lib_executable(carma-miniapp
  carma
  ${carma_deps})

add_lib_executable(carma-statistics
  carma)

add_lib_executable(temp-miniapp
  carma
  ${carma_deps})

target_link_libraries(carma-miniapp semiprof)
target_link_libraries(carma-statistics semiprof)
target_link_libraries(temp-miniapp semiprof)
