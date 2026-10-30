#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "../utils/cosma_utils.hpp"

// Checks that the cached communicator of a context that outlives MPI_Finalize
// is released during MPI_Finalize. This has to be observed from within
// MPI_Finalize, hence the standalone main instead of gtest.

namespace {
bool comm_released = false;

// Delete callbacks on MPI_COMM_SELF run in reverse order of attachment.
// Attached before any COSMA call, this one runs after COSMA's finalize hook.
int check_comm_released(MPI_Comm, int, void *attr_val, void *) {
    auto *ctx = static_cast<cosma_context<double> *>(attr_val);
    comm_released = ctx->get_cosma_comm() == nullptr;
    return MPI_SUCCESS;
}

Strategy make_strategy() {
    constexpr int m = 100;
    constexpr int n = 100;
    constexpr int k = 100;
    constexpr int P = 4;
    std::string step_types = "pp";
    std::string dims = "mn";
    std::vector<int> divs = {2, 2};
    return Strategy(m, n, k, P, divs, dims, step_types);
}
} // namespace

int main(int argc, char **argv) {
    // outlives MPI_Finalize, like the global context does
    context<double> ctx;

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ctx = make_context<double>();

    int keyval;
    MPI_Comm_create_keyval(
        MPI_COMM_NULL_COPY_FN, &check_comm_released, &keyval, nullptr);
    MPI_Comm_set_attr(MPI_COMM_SELF, keyval, ctx.get());
    MPI_Comm_free_keyval(&keyval);

    auto strategy = make_strategy();

    {
        // destroyed before MPI_Finalize: its hook has to detach
        auto early_ctx = make_context<double>();
        if (!test_cosma<double>(strategy, early_ctx)) {
            std::cerr << "early_ctx: wrong multiply result" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (!test_cosma<double>(strategy, ctx)) {
        std::cerr << "ctx: wrong multiply result" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (ctx->get_cosma_comm() == nullptr) {
        std::cerr << "rank " << rank << ": no communicator cached before "
                  << "MPI_Finalize" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();

    if (!comm_released) {
        std::cerr << "rank " << rank << ": cached communicator not released "
                  << "during MPI_Finalize" << std::endl;
        return 1;
    }
    return 0;
}
