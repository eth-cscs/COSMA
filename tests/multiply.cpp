#include "../utils/cosma_utils.hpp"
#include <initializer_list>

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
    std::vector<int> divs;
    std::string dims = "";
    std::string step_types = "";

    multiply_state() = default;

    multiply_state(int mm, int nn, int kk, int PP,
                   std::vector<int> divisors,
                   std::string dim,
                   std::string steps)
        : m(mm)
        , n(nn)
        , k(kk)
        , P(PP)
        , divs(divisors)
        , dims(dim)
        , step_types(steps)
    {}

    multiply_state(int mm, int nn, int kk, int PP)
        : m(mm)
        , n(nn)
        , k(kk)
        , P(PP)
    {}

    static int& get_test_counter() {
        static int test_counter = 0;
        return test_counter;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const multiply_state &obj) {
        return os << "(m, n, k) = (" << obj.m << ", " << obj.n << ", " << obj.k
                  << ")" << std::endl
                  << "Number of ranks: " << obj.P << std::endl
                  << "Strategy: " << obj.dims << ", " << obj.step_types << std::endl;
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
    double epsilon = 1e-8;
    auto state = GetParam();

    MPI_Barrier(MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m = state.m;
    int n = state.n;
    int k = state.k;
    int P = state.P;
    MPI_Comm comm = subcommunicator(P);

    if (rank >= P) {
        ++multiply_state::get_test_counter();
        ++multiply_state::get_test_counter();
    }

    if (rank < P) {
        Strategy::min_dim_size = 32;
        Strategy strategy(m, n, k, P, state.divs, state.dims, state.step_types);
        if (rank == 0) {
            std::cout << "Strategy = " << strategy << std::endl;
        }

        // first run without overlapping communication and computation
        bool no_overlap = test_cosma<double>(strategy, ctx, comm, epsilon,
                                             multiply_state::get_test_counter());
        ++multiply_state::get_test_counter();

        EXPECT_TRUE(no_overlap);

        // wait for no-overlap to finish
        MPI_Barrier(comm);

        // then run with the overlap of communication and computation
        strategy.enable_overlapping_comm_and_comp();
        bool with_overlap = test_cosma<double>(strategy, ctx, comm, epsilon, multiply_state::get_test_counter());
        ++multiply_state::get_test_counter();
        EXPECT_TRUE(with_overlap);

        MPI_Comm_free(&comm);
    }
}

std::vector<multiply_state> generate_tests() {
    return {
        multiply_state(4, 4, 4, 1),
        multiply_state(3, 4, 5, 1),

        // strategy: pm2,sm2,pn2
        multiply_state(8, 4, 2, 4,
                    {2, 2, 2}, // divisors
                    "mmn", // split dimensions
                    "psp" // step types
        ),
        multiply_state(8, 4, 2, 4),

        multiply_state(4, 4, 4, 2, 
                       {2},// divisors
                       "m", // split dimensions
                       "p" // step types
        ),
        multiply_state(4, 4, 4, 2),

        multiply_state(4, 4, 4, 4, 
                       {2, 2, 2},// divisors
                       "mnn", // split dimensions
                       "spp" // step types
        ),

        // -strategy: sm2,pn2,pn2
        multiply_state(4, 4, 4, 4, 
                       {2, 2, 2},// divisors
                       "mnn", // split dimensions
                       "spp" // step types
        ),

        multiply_state(30, 35, 40, 4),
        // strategy: sm2,sm2,pn2
        multiply_state(8, 8, 2, 2,
            {2, 2, 2}, // divisors
            "mmn", // split dimensions
            "ssp" // step types
        ),
        multiply_state(8, 8, 2, 2),

        // strategy: pm2,pm2
        multiply_state(16, 4, 4, 4,
                       {2, 2}, // divisors
                       "mm", // split dimensions
                       "pp" // step types
        ),
        multiply_state(16, 4, 4, 4),

        // startegy: sk2,pm3
        multiply_state(20, 20, 20, 3,
                       {2, 3}, // divisors
                       "km", // split dimensions
                       "sp" // step types
        ),
        multiply_state(20, 20, 20, 3),

        // strategy: pm2,pn2,pk2,pm2
        multiply_state(16, 16, 16, 16,
                       {2, 2, 2, 2}, // divisors
                       "mnkm", // split dimensions
                       "pppp" // step types
        ),
        multiply_state(16, 16, 16, 16),

        // strategy: sm2,sn2,pk2,pm2
        multiply_state(20, 30, 25, 4,
                       {2, 2, 2, 2}, // divisors
                       "mnkm", // split dimensions
                       "sspp" // step types
        ),
        multiply_state(20, 30, 25, 4),

        // strategy: sm2,pn2,sk2,pm5
        multiply_state(100, 100, 100, 10,
                       {2, 2, 2, 5}, // divisors
                       "mnkm", // split dimensions
                       "spsp" // step types
        ),
        multiply_state(100, 100, 100, 10),

        // strategy: sm2,pn2,sk2,pm2
        multiply_state(4, 4, 5, 4, 
                       {2, 2, 2, 2},  // divisors
                       "mnkm", // split dimensions
                       "spsp" // step types
        ),
        multiply_state(4, 4, 5, 4),

        // strategy: pm2,pn2,pk3
        multiply_state(100, 100, 100, 12,
                       {2, 2, 3}, // divisors
                       "mnk", // split dimensions
                       "ppp" // step types
        ),
        multiply_state(100, 100, 100, 12),

        multiply_state(100, 100, 100, 4),

        // strategy: pm7
        multiply_state(100, 100, 100, 7,
                       {7}, // divisors
                       "m", // split dimensions
                       "p" // step types
        ),
        multiply_state(100, 100, 100, 7),

        // strategy: sm2,pn2,sk2,pm2,sn2,pk2
        multiply_state(100, 100, 100, 8,
                       {2, 2, 2, 2, 2, 2}, // divisors
                       "mnkmnk", // split dimensions
                       "spspsp" // step types
        ),
        multiply_state(100, 100, 100, 8),

        // strategy: pm2,pk2
        multiply_state(31, 32, 33, 4,
                       {2, 2}, // divisors
                       "mk", // split dimensions
                       "pp" // step types
        ),
        // strategy: pm2,pk2
        multiply_state(100, 100, 100, 4,
                       {2, 2}, // divisors
                       "mk", // split dimensions
                       "pp" // step types
        ),
        // strategy: pm2,pk2,pn2
        multiply_state(100, 100, 100, 8,
                       {2, 2}, // divisors
                       "mk", // split dimensions
                       "pp" // step types
        ),

        // strategy: sm2,sk2,sn2,pn2,pm2,pk2
        multiply_state(100, 100, 100, 8,
                       {2, 2, 2, 2, 2, 2}, // divisors
                       "mknnmk", // split dimensions
                       "sssppp" // step types
        ),

        // strategy: sk2,pm2,sn2,pk2,sm2,pn2
        multiply_state(100, 100, 100, 8,
                       {2, 2, 2, 2, 2, 2}, // divisors
                       "kmnkmn", // split dimensions
                       "spspsp" // step types
        ),

        // strategy: sk3,sm3,sn3,pk2,pn2,pm2
        multiply_state(200, 200, 200, 8,
                       {3, 3, 3, 2, 2, 2}, // divisors
                       "kmnknm", // split dimensions
                       "sssppp" // step types
        ),
        multiply_state(200, 200, 200, 8),

        // strategy: sm3,pn2,sk3,pm2,sn3,pk2
        multiply_state(200, 200, 200, 8,
                       {3, 2, 3, 2, 3, 2}, // divisors
                       "mnkmnk", // split dimensions
                       "spspsp" // step types
        ), 

        multiply_state(512, 32, 736, 8,
                       {2, 2, 2},
                       "kmk",
                       "ppp"
        )
    };
};

INSTANTIATE_TEST_CASE_P(
    Default,
    MultiplyTestWithParams,
    testing::ValuesIn(generate_tests()));
