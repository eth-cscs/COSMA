#include <cosma/gpu/utils.hpp>

void cosma::gpu::free_nccl_comm(ncclComm_t nccl_comm) {
    auto status = ncclCommDestroy(nccl_comm);
    check_nccl_status(status);
}

void cosma::gpu::check_nccl_status(ncclResult_t result) {
    if (result != ncclSuccess) {
        std::cerr << "[NCCL ERROR]: " << ncclGetErrorString(result) << std::endl;
        throw(std::runtime_error("NCCL ERROR"));
    }
}

void cosma::gpu::check_runtime_status(runtime_api::StatusType status) {
    if(status !=  runtime_api::status::Success) {
        std::cerr << "error: GPU API call : "
        << runtime_api::get_error_string(status) << std::endl;
        throw(std::runtime_error("GPU ERROR"));
    }
}

ncclComm_t cosma::gpu::mpi_to_nccl_comm(MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) {
        return nullptr;
    }
    int my_rank, n_ranks;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_ranks);

    ncclUniqueId id;
    if (my_rank == 0) {
        auto status = ncclGetUniqueId(&id);
        check_nccl_status(status);
    }

    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);

    ncclComm_t nccl_comm;
    auto status = ncclCommInitRank(&nccl_comm, n_ranks, id, my_rank);
    check_nccl_status(status);

    return nccl_comm;
}
