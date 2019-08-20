# Sets COSMA_WITH_OPENMPI to ON if OpenMPI was found and to OFF otherwise.
#
function(check_for_openmpi)
  execute_process(COMMAND mpirun --version OUTPUT_VARIABLE MPIRUN_OUTPUT)
  string(FIND "${MPIRUN_OUTPUT}" "Open MPI" OMPI_POS)
  if(OMPI_POS STREQUAL "-1")
    set(COSMA_WITH_OPENMPI OFF PARENT_SCOPE)
  else()
    set(COSMA_WITH_OPENMPI ON PARENT_SCOPE)
  endif()
endfunction()
