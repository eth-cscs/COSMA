#include <cosma_run.hpp>

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
    MPI_Group_excl(
        group, exclude_ranks.size(), exclude_ranks.data(), &newcomm_group);
    // create reduced communicator
    MPI_Comm_create_group(comm, newcomm_group, 0, &newcomm);

    MPI_Group_free(&group);
    MPI_Group_free(&newcomm_group);

    return newcomm;
}

struct multiply_state {
    int m = 10;
    int n = 10;
    int k = 10;
    int P = 2;
    std::string steps = "";

    multiply_state() = default;

    multiply_state(int mm, int nn, int kk, int PP, std::string ssteps)
        : m(mm)
        , n(nn)
        , k(kk)
        , P(PP)
        , steps(ssteps) {}

    multiply_state(int mm, int nn, int kk, int PP)
        : m(mm)
        , n(nn)
        , k(kk)
        , P(PP)
        , steps("") {}

    friend std::ostream &operator<<(std::ostream &os,
                                    const multiply_state &obj) {
        return os << "(m, n, k) = (" << obj.m << ", " << obj.n << ", " << obj.k
                  << ")\n"
                  << "Number of ranks: " << obj.P << "\n"
                  << "Strategy: " << obj.steps << "\n";
    }
};

struct MultiplyTest : testing::Test {
    cosma::context<double> ctx;
    std::unique_ptr<multiply_state> state;

    MultiplyTest() {
        ctx = cosma::make_context<double>();
        state = std::make_unique<multiply_state>();
    }
};

struct MultiplyTestWithParams : MultiplyTest,
                                testing::WithParamInterface<multiply_state> {
    MultiplyTestWithParams() = default;
};

TEST_P(MultiplyTestWithParams, multiply) {
    auto state = GetParam();

    MPI_Barrier(MPI_COMM_WORLD);

    int total_P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &total_P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m = state.m;
    int n = state.n;
    int k = state.k;
    int P = state.P;
    MPI_Comm comm = subcommunicator(P);

    if (rank < P) {
        std::string steps = state.steps;
        Strategy strategy(m, n, k, P, steps);

        if (rank == 0) {
            std::cout << "Strategy = " << strategy << std::endl;
        }

        // first run without overlapping communication and computation
        bool no_overlap = run<double>(strategy, ctx, comm, false);
        EXPECT_TRUE(no_overlap);

        // wait for no-overlap to finish
        MPI_Barrier(comm);

        // then run with the overlap of communication and computation
        bool with_overlap = run<double>(strategy, ctx, comm, true);
        EXPECT_TRUE(with_overlap);

        MPI_Comm_free(&comm);
    }
}

INSTANTIATE_TEST_CASE_P(
    Default,
    MultiplyTestWithParams,
    testing::Values(
        multiply_state{4, 4, 4, 1},
        multiply_state{3, 4, 5, 1},

        multiply_state{4, 4, 4, 4, "-s sm2,pn2,pn2"},

        multiply_state{30, 35, 40, 4},

        multiply_state{8, 4, 2, 4, "-s pm2,sm2,pn2"},
        multiply_state{8, 4, 2, 4},

        multiply_state{8, 8, 2, 2, "-s sm2,sm2,pn2"},
        multiply_state{8, 8, 2, 2},

        multiply_state{16, 4, 4, 4, "-s pm2,pm2"},
        multiply_state{16, 4, 4, 4},

        multiply_state{20, 20, 20, 3, "-s sk2,pm3"},
        multiply_state{20, 20, 20, 3},

        multiply_state{16, 16, 16, 16, "-s pm2,pn2,pk2,pm2"},
        multiply_state{16, 16, 16, 16},

        multiply_state{20, 30, 25, 4, "-s sm2,sn2,pk2,pm2"},
        multiply_state{20, 30, 25, 4},

        multiply_state{100, 100, 100, 10, "-s sm2,pn2,sk2,pm5"},
        multiply_state{100, 100, 100, 10},

        multiply_state{4, 4, 5, 4, "-s sm2,pn2,sk2,pm2"},
        multiply_state{4, 4, 5, 4},

        multiply_state{100, 100, 100, 12, "-s pm2,pn2,pk3"},
        multiply_state{100, 100, 100, 12},

        multiply_state{100, 100, 100, 4},

        multiply_state{100, 100, 100, 7, "-s pm7"},
        multiply_state{100, 100, 100, 7},

        multiply_state{100, 100, 100, 8, "-s sm2,pn2,sk2,pm2,sn2,pk2"},
        multiply_state{100, 100, 100, 8},

        multiply_state{100, 100, 100, 8, "-s sm2,sk2,sn2,pn2,pm2,pk2"},

        multiply_state{100, 100, 100, 8, "-s sk2,pm2,sn2,pk2,sm2,pn2"},

        multiply_state{200, 200, 200, 8, "-s sk3,sm3,sn3,pk2,pn2,pm2"},
        multiply_state{200, 200, 200, 8},

        multiply_state{200, 200, 200, 8, "-s sm3,pn2,sk3,pm2,sn3,pk2"}));
