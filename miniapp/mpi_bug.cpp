#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    double* pointer;
    int size = 500;

    MPI_Alloc_mem(size * sizeof(double), MPI_INFO_NULL, &pointer);

    MPI_Win win;
    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    void* disp_unit;
    int flag;
    void* win_size;
    MPI_Win_get_attr(win, MPI_WIN_DISP_UNIT, &disp_unit, &flag);
    MPI_Win_get_attr(win, MPI_WIN_SIZE, &win_size, &flag);
    std::cout << "before disp_unit = " << *(int *)disp_unit << std::endl;
    std::cout << "before size = " << *(int *)win_size << std::endl;
    MPI_Win_attach(win, pointer, size);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_get_attr(win, MPI_WIN_DISP_UNIT, &win_size, &flag);
    MPI_Win_get_attr(win, MPI_WIN_SIZE, &win_size, &flag);
    std::cout << "after disp_unit = " << *(int *)disp_unit << std::endl;
    std::cout << "after size = " << *(int *)win_size << std::endl;

    MPI_Win_detach(win, pointer);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win);

    MPI_Free_mem(pointer);

    MPI_Finalize();
    return 0;
}
