#include "../utils/pxgemm_utils.hpp"
#include <cosma/strategy.hpp>

#include <gtest/gtest.h>
#include <gtest_mpi/gtest_mpi.hpp>

MPI_Comm subcommunicator(int new_P, MPI_Comm comm = MPI_COMM_WORLD) {
    // original size
    int P;
    MPI_Comm_size(comm, &P);

    // original group
    MPI_Group group;
    MPI_Comm_group(comm, &group);

    // new comm and new group
    MPI_Comm newcomm;
    MPI_Group newcomm_group;

    // ranks to exclude
    std::vector<int> exclude_ranks;
    for (int i = new_P; i < P; ++i) {
        exclude_ranks.push_back(i);
    }
    // create reduced group
    MPI_Group_excl(group, exclude_ranks.size(), exclude_ranks.data(), &newcomm_group);
    // create reduced communicator
    MPI_Comm_create_group(comm, newcomm_group, 0, &newcomm);

    MPI_Group_free(&group);
    MPI_Group_free(&newcomm_group);

    return newcomm;
}

struct PdgemmTest : testing::Test {
    std::unique_ptr<cosma::pxgemm_params<double>> state;

    PdgemmTest() {
        state = std::make_unique<cosma::pxgemm_params<double>>();
    }
};

struct PdgemmTestWithParams : PdgemmTest,
                              testing::WithParamInterface<cosma::pxgemm_params<double>> {
    PdgemmTestWithParams() = default;
};

TEST_P(PdgemmTestWithParams, pdgemm) {
    auto state = GetParam();

    MPI_Barrier(MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm = subcommunicator(state.P, MPI_COMM_WORLD);

    if (rank < state.P) {
        if (rank == 0) {
            std::cout << state << std::endl;
        }

        cosma::Strategy::min_dim_size = 32;
        bool correct = test_pdgemm(state, comm);

        EXPECT_TRUE(correct);
        MPI_Comm_free(&comm);
    }
};

INSTANTIATE_TEST_CASE_P(
    Default,
    PdgemmTestWithParams,
    testing::Values(
        // alpha = 1.0, beta = 0.0
        // single process
        cosma::pxgemm_params<double>{10, 10, 10, 2, 2, 2, 1, 1, 'N', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{10, 11, 13, 2, 2, 2, 1, 1, 'N', 'N', 1.0, 0.0},

        // default values of alpha and beta
        cosma::pxgemm_params<double>{10, 10, 10, 2, 2, 2, 2, 2, 'N', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{5, 5, 5, 2, 2, 2, 2, 2, 'N', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{5, 5, 5, 2, 2, 2, 2, 2, 'T', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{8, 4, 8, 2, 2, 2, 3, 2, 'N', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{8, 4, 8, 2, 2, 2, 3, 2, 'T', 'N', 1.0, 0.0},

        // different values of alpha and beta
        cosma::pxgemm_params<double>{10, 12, 12, 2, 2, 2, 2, 2, 'T', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{10, 11, 12, 3, 2, 3, 3, 2, 'T', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{10, 11, 12, 3, 2, 3, 3, 2, 'T', 'N', 1.0, 1.0},
        cosma::pxgemm_params<double>{10, 11, 12, 3, 2, 3, 3, 2, 'T', 'N', 0.0, 0.0},
        cosma::pxgemm_params<double>{10, 11, 12, 3, 2, 3, 3, 2, 'T', 'N', 0.0, 1.0},
        cosma::pxgemm_params<double>{10, 11, 12, 3, 2, 3, 3, 2, 'T', 'N', 0.5, 0.5},

        // alpha = 0.5, beta = 0.0
        cosma::pxgemm_params<double>{10, 10, 10, 2, 2, 2, 2, 2, 'N', 'N', 0.5, 0.0},
        cosma::pxgemm_params<double>{5, 5, 5, 2, 2, 2, 2, 2, 'N', 'N', 0.5, 0.0},
        cosma::pxgemm_params<double>{5, 5, 5, 2, 2, 2, 2, 2, 'T', 'N', 0.5, 0.0},
        cosma::pxgemm_params<double>{8, 4, 8, 2, 2, 2, 3, 2, 'N', 'N', 0.5, 0.0},
        cosma::pxgemm_params<double>{8, 4, 8, 2, 2, 2, 3, 2, 'T', 'N', 0.5, 0.0},

        // too many resources
        cosma::pxgemm_params<double>{16, 16, 96, 32, 32, 32, 2, 8, 'T', 'N', 0.5, 0.5},
        cosma::pxgemm_params<double>{13, 13, 448, 13, 13, 13, 2, 7, 'T', 'N', 0.5, 0.5},
        cosma::pxgemm_params<double>{13, 13, 448, 13, 13, 13, 2, 7, 'N', 'N', 1.0, 0.5},

        cosma::pxgemm_params<double>{26, 13, 448, 13, 13, 13, 2, 7, 'T', 'N', 1.0, 0.5},
        // detailed pdgemm call
        cosma::pxgemm_params<double>{
            // matrix dimensions
            1280, 1280, // matrix A
            1280, 1280, // matrix B
            1280, 1280, // matrix C

            // block sizes
            32, 32, // matrix A
            32, 32, // matrix B
            32, 32, // matrix C

            // submatrices ij
            1, 545, // matrix A
            513, 545, // matrix B
            1, 513, // matrix C

            // problem size
            512, 32, 736,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 1.0,

            // leading dims
            640, 640, 640,

            // proc grid
            2, 4, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        }
    ));

