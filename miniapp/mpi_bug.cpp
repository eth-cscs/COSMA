#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    double* pointer;
    int size = 500;

    MPI_Alloc_mem(size * sizeof(double), MPI_INFO_NULL, &pointer);

    MPI_Win win;
    int error_creation = MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    void* disp_unit;
    int flag;
    int error_get_attr = MPI_Win_get_attr(win, MPI_WIN_DISP_UNIT, &disp_unit, &flag);
    std::cout << "flag = " << flag << std::endl;
    std::cout << "disp_unit = " << *(int *)disp_unit << std::endl;
    int error_attach = MPI_Win_attach(win, pointer, size);

    MPI_Barrier(MPI_COMM_WORLD);


    bool no_errors = (error_creation == MPI_SUCCESS) && (error_attach == MPI_SUCCESS)
                    && (error_get_attr == MPI_SUCCESS);
    std::cout << "No errors? " << no_errors << std::endl;

    MPI_Win_detach(win, pointer);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win);

    MPI_Free_mem(pointer);

    MPI_Finalize();
    return 0;
}
