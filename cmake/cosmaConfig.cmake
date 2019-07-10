if(NOT TARGET cosma::cosma)
  include(CMakeFindDependencyMacro)
  find_dependency(MPI)
  find_dependency(MKL)

  include("${CMAKE_CURRENT_LIST_DIR}/cosmaTargets.cmake")
endif()
