#include "../utils/pxgemm_utils.hpp"
#include <cosma/strategy.hpp>

#include <gtest/gtest.h>
#include <gtest_mpi/gtest_mpi.hpp>
#include <vector>
#include <iostream>

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

        // cosma::Strategy::min_dim_size = 32;
        bool correct = test_pxgemm<double>(state, comm);
        EXPECT_TRUE(correct);

        MPI_Comm_free(&comm);
    }
};

INSTANTIATE_TEST_CASE_P(
    Default,
    PdgemmTestWithParams,
    testing::Values(
        // edge cases, which are allowed by the standard (m, n or k can be 0)
        cosma::pxgemm_params<double>{
            // matrix dimensions
            10, 10, // matrix A
            10, 10, // matrix B
            10, 10, // matrix C

            // block sizes
            2, 2, // matrix A
            2, 2, // matrix B
            2, 2, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            0, 5, 5,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 1.0,

            // leading dims
            10, 10, 10,

            // proc grid
            2, 2, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            10, 10, // matrix A
            10, 10, // matrix B
            10, 10, // matrix C

            // block sizes
            2, 2, // matrix A
            2, 2, // matrix B
            2, 2, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            5, 0, 5,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 1.0,

            // leading dims
            10, 10, 10,

            // proc grid
            2, 2, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            10, 10, // matrix A
            10, 10, // matrix B
            10, 10, // matrix C

            // block sizes
            2, 2, // matrix A
            2, 2, // matrix B
            2, 2, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            0, 0, 0,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 1.0,

            // leading dims
            10, 10, 10,

            // proc grid
            2, 2, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            10, 10, // matrix A
            10, 10, // matrix B
            10, 10, // matrix C

            // block sizes
            2, 2, // matrix A
            2, 2, // matrix B
            2, 2, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            10, 0, 0,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 1.0,

            // leading dims
            10, 10, 10,

            // proc grid
            2, 2, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        // scaling matrix C checking (k=0)
        cosma::pxgemm_params<double>{
            // matrix dimensions
            10, 10, // matrix A
            10, 10, // matrix B
            10, 10, // matrix C

            // block sizes
            2, 2, // matrix A
            2, 2, // matrix B
            2, 2, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            10, 10, 0,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 1.2,

            // leading dims
            10, 10, 10,

            // proc grid
            2, 2, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        // scaling matrix C checking (k=0, irregular)
        cosma::pxgemm_params<double>{
            // matrix dimensions
            23, 34, // matrix A
            34, 53, // matrix B
            23, 53, // matrix C

            // block sizes
            2, 3, // matrix A
            4, 5, // matrix B
            5, 7, // matrix C

            // submatrices ij
            1, 2, // matrix A
            2, 3, // matrix B
            3, 4, // matrix C

            // problem size
            7, 11, 0,

            // transpose flags
            'N', 'N',

            // scaling flags
            1.0, 1.2,

            // leading dims
            53, 54, 55,

            // proc grid
            2, 3, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{4, 4, 4, 2, 2, 2, 2, 1, 'T', 'N', 1.0, 0.0},

        // scaling matrix C checking (alpha = 0)
        cosma::pxgemm_params<double>{
            // matrix dimensions
            10, 10, // matrix A
            10, 10, // matrix B
            10, 10, // matrix C

            // block sizes
            2, 2, // matrix A
            2, 2, // matrix B
            2, 2, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            10, 10, 10,

            // transpose flags
            'N', 'T',

            // scaling flags
            0.0, 1.2,

            // leading dims
            10, 10, 10,

            // proc grid
            2, 2, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        // scaling matrix C checking (alpha = 0, irregular)
        cosma::pxgemm_params<double>{
            // matrix dimensions
            23, 34, // matrix A
            34, 53, // matrix B
            23, 53, // matrix C

            // block sizes
            2, 3, // matrix A
            4, 5, // matrix B
            5, 7, // matrix C

            // submatrices ij
            1, 2, // matrix A
            2, 3, // matrix B
            3, 4, // matrix C

            // problem size
            7, 11, 5,

            // transpose flags
            'N', 'N',

            // scaling flags
            0.0, 1.2,

            // leading dims
            53, 54, 55,

            // proc grid
            2, 3, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

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
        cosma::pxgemm_params<double>{2, 2, 8, 2, 2, 4, 2, 1, 'T', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{4, 4, 24, 4, 4, 8, 2, 1, 'T', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{16, 16, 96, 16, 16, 32, 2, 1, 'T', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{16, 16, 96, 32, 32, 32, 2, 8, 'T', 'N', 0.5, 0.5},
        cosma::pxgemm_params<double>{13, 13, 448, 13, 13, 13, 2, 7, 'T', 'N', 0.5, 0.5},
        cosma::pxgemm_params<double>{13, 13, 448, 13, 13, 13, 2, 7, 'N', 'N', 1.0, 0.5},

        cosma::pxgemm_params<double>{3, 3, 7, 3, 3, 3, 1, 1, 'T', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{5, 5, 11, 5, 5, 5, 2, 1, 'T', 'N', 1.0, 0.0},
        cosma::pxgemm_params<double>{26, 13, 448, 13, 13, 13, 2, 1, 'T', 'N', 1.0, 0.5},
        cosma::pxgemm_params<double>{26, 13, 448, 13, 13, 13, 2, 7, 'T', 'N', 1.0, 0.5},

        // adapt strategy to scalapack grid when P = 1
        cosma::pxgemm_params<double>{
            // matrix dimensions
            1280, 128, // matrix A
            1280, 128, // matrix B
            128, 128, // matrix C

            // block sizes
            32, 32, // matrix A
            32, 32, // matrix B
            32, 32, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            128, 128, 1280,

            // transpose flags
            'T', 'N',

            // scaling flags
            1.0, 0.0,

            // leading dims
            1280, 1280, 128,

            // proc grid
            1, 1, 'C',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        // detailed pdgemm call with ia, ja = 1
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
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            512, 32, 736,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 0.0,

            // leading dims
            640, 640, 640,

            // proc grid
            2, 4, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

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
            1.0, 0.0,

            // leading dims
            640, 640, 640,

            // proc grid
            2, 4, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

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
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            1000, 10, // matrix A
            1000, 10, // matrix B
            10, 10, // matrix C

            // block sizes
            128, 128, // matrix A
            128, 128, // matrix B
            128, 128, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            10, 10, 1000,

            // transpose flags
            'T', 'N',

            // scaling flags
            1.0, 0.0,

            // leading dims
            512, 512, 10,

            // proc grid
            2, 2, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },


        cosma::pxgemm_params<double>{
            // matrix dimensions
            1824, 128, // matrix A
            1824, 128, // matrix B
            128, 128, // matrix C

            // block sizes
            32, 32, // matrix A
            32, 32, // matrix B
            32, 32, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            128, 128, 1824,

            // transpose flags
            'T', 'N',

            // scaling flags
            1.0, 0.0,

            // leading dims
            928, 928, 64,

            // proc grid
            2, 4, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            43417, 217, // matrix A
            43417, 217, // matrix B
            217, 217, // matrix C

            // block sizes
            169, 108, // matrix A
            169, 108, // matrix B
            108, 108, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            217, 217, 43417,

            // transpose flags
            'T', 'N',

            // scaling flags
            1.0, 0.0,

            // leading dims
            54272, 54272, 1088,

            // proc grid
            8, 1, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            43176, 217, // matrix A
            43176, 217, // matrix B
            2176, 217, // matrix C

            // block sizes
            1696, 108, // matrix A
            1696, 108, // matrix B
            1088, 108, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            217, 217, 43176,

            // transpose flags
            'T', 'N',

            // scaling flags
            1.0, 0.0,

            // leading dims
            54272, 54272, 1088,

            // proc grid
            8, 1, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

	// CP2K runs from H2O-sos-mp2-lr
        cosma::pxgemm_params<double>{
            // matrix dimensions
            23, 23, // matrix A
            23, 23, // matrix B
            23, 23, // matrix C

            // block sizes
            12, 12, // matrix A
            12, 12, // matrix B
            12, 12, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            23, 23, 4,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 0.0,

            // leading dims
            12, 12, 12,

            // proc grid
            2, 1, 'C',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            23, 23, // matrix A
            23, 23, // matrix B
            23, 23, // matrix C

            // block sizes
            12, 12, // matrix A
            12, 12, // matrix B
            12, 12, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            23, 23, 4,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, -1.0,

            // leading dims
            12, 12, 12,

            // proc grid
            2, 1, 'C',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            83, 83, // matrix A
            83, 83, // matrix B
            83, 83, // matrix C

            // block sizes
            32, 32, // matrix A
            32, 32, // matrix B
            32, 32, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            83, 83, 83,

            // transpose flags
            'N', 'T',

            // scaling flags
            1.0, 0.0,

            // leading dims
            83, 83, 83,

            // proc grid
            1, 1, 'C',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            83, 83, // matrix A
            83, 77, // matrix B
            83, 77, // matrix C

            // block sizes
            32, 32, // matrix A
            32, 32, // matrix B
            32, 32, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            83, 77, 83,

            // transpose flags
            'T', 'N',

            // scaling flags
            1.0, 0.0,

            // leading dims
            83, 83, 83,

            // proc grid
            1, 1, 'C',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        },

        cosma::pxgemm_params<double>{
            // matrix dimensions
            83, 83, // matrix A
            83, 77, // matrix B
            83, 77, // matrix C

            // block sizes
            32, 32, // matrix A
            32, 32, // matrix B
            32, 32, // matrix C

            // submatrices ij
            1, 1, // matrix A
            1, 1, // matrix B
            1, 1, // matrix C

            // problem size
            83, 77, 83,

            // transpose flags
            'T', 'N',

            // scaling flags
            1.0, 0.0,

            // leading dims
            83, 83, 83,

            // proc grid
            1, 1, 'R',

            // proc srcs
            0, 0, // matrix A
            0, 0, // matrix B
            0, 0  // matrix C
        }
    )
);

