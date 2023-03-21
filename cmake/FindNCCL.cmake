include(FindPackageHandleStandardArgs)

find_path(COSMA_NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS
  ${NCCL_ROOT}
  ENV NCCLROOT
)

find_library(COSMA_NCCL_LIBRARIES
  NAMES nccl nccl_static
  HINTS
  ${NCCL_ROOT}
  ENV NCCLROOT
)

find_package_handle_standard_args(NCCL DEFAULT_MSG COSMA_NCCL_INCLUDE_DIRS COSMA_NCCL_LIBRARIES)

if (NCCL_FOUND AND NOT TARGET cosma::nccl)
  add_library(cosma::nccl INTERFACE IMPORTED)
  set_target_properties(cosma::nccl
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${COSMA_NCCL_INCLUDE_DIRS}
    INTERFACE_LINK_LIBRARIES ${COSMA_NCCL_LIBRARIES})
endif()
