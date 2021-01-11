# Appends the --oversubscribe flag if OpenMPI.
#
function(adjust_mpiexec_flags)
    execute_process(COMMAND mpirun --version OUTPUT_VARIABLE MPIRUN_OUTPUT)
    string(FIND "${MPIRUN_OUTPUT}" "Open MPI" OMPI_POS)
    if(NOT OMPI_POS STREQUAL "-1")
        set(MPIEXEC_PREFLAGS "--oversubscribe;${MPIEXEC_PREFLAGS}" CACHE STRING "These flags will be directly before the executable that is being run by mpiexec." FORCE)
        set(MPI_TYPE "ompi" PARENT_SCOPE)
    else()
        set(MPI_TYPE "mpich" PARENT_SCOPE)
    endif()
endfunction()
