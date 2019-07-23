#include <cosma_pdgemm_run.hpp>

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

struct pdgemm_state {
    int m = 10;
    int n = 10;
    int k = 10;
    int bm = 2;
    int bn = 2;
    int bk = 2;
    int p_rows = 2;
    int p_cols = 1;
    int P = 2;
    char trans_a = 'N';
    char trans_b = 'N';
    double alpha = 1.0;
    double beta = 0.0;

    pdgemm_state() = default;

    pdgemm_state(int m, int n, int k,
                   int bm, int bn, int bk,
                   int p_rows, int p_cols,
                   char trans_a, char trans_b):
        m(m), n(n), k(k), bm(bm), bn(bn), bk(bk),
        p_rows(p_rows), p_cols(p_cols), P(p_rows*p_cols),
        trans_a(trans_a), trans_b(trans_b) {}

    pdgemm_state(int m, int n, int k,
                   int bm, int bn, int bk,
                   int p_rows, int p_cols,
                   char trans_a, char trans_b,
                   double alpha, double beta):
        m(m), n(n), k(k), bm(bm), bn(bn), bk(bk),
        p_rows(p_rows), p_cols(p_cols), P(p_rows*p_cols),
        trans_a(trans_a), trans_b(trans_b),
        alpha(alpha), beta(beta) {}

    friend std::ostream &operator<<(std::ostream &os,
                                    const pdgemm_state &obj) {
        return os << "(m, n, k) = (" << obj.m << ", " << obj.n << ", " << obj.k
                  << ")\n"
                  << "(alpha, beta) = (" << obj.alpha << ", " << obj.beta << ")\n"
                  << "Number of ranks: " << obj.P << "\n"
                  << "Process grid: (" << obj.p_rows << ", " << obj.p_cols << ")" << "\n"
                  << "Block sizes = (" << obj.bm << ", " << obj.bn << ", " << obj.bk << ")" << "\n"
                  << "Transpose flags = (" << obj.trans_a << ", " << obj.trans_b << "\n";
    }
};

struct PdgemmTest : testing::Test {
    std::unique_ptr<pdgemm_state> state;

    PdgemmTest() {
        state = std::make_unique<pdgemm_state>();
    }
};

struct PdgemmTestWithParams : PdgemmTest,
                                testing::WithParamInterface<pdgemm_state> {
    PdgemmTestWithParams() = default;
};

TEST_P(PdgemmTestWithParams, pdgemm) {
    auto state = GetParam();

    MPI_Barrier(MPI_COMM_WORLD);

    int total_P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &total_P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m = state.m;
    int n = state.n;
    int k = state.k;
    int P = state.P;
    int p = state.p_rows;
    int q = state.p_cols;
    int bm = state.bm;
    int bn = state.bn;
    int bk = state.bk;
    double alpha = state.alpha;
    double beta = state.beta;

    MPI_Comm comm = subcommunicator(P);

    if (rank < P) {
        if (rank == 0) {
            std::cout << state << std::endl;
        }

        bool correct = test_pdgemm(m, n, k, bm, bn, bk,
            1, 1, 1, state.trans_a, state.trans_b, p, q,
            alpha, beta,
            rank, comm);
        EXPECT_TRUE(correct);

        MPI_Comm_free(&comm);
    }
};

INSTANTIATE_TEST_CASE_P(
    Default,
    PdgemmTestWithParams,
    testing::Values(
        // alpha = 1.0, beta = 0.0
        pdgemm_state{10, 10, 10, 2, 2, 2, 2, 2, 'N', 'N'},
        pdgemm_state{5, 5, 5, 2, 2, 2, 2, 2, 'N', 'N'},
        pdgemm_state{5, 5, 5, 2, 2, 2, 2, 2, 'T', 'N'},
        pdgemm_state{8, 4, 8, 2, 2, 2, 3, 2, 'N', 'N'},
        pdgemm_state{8, 4, 8, 2, 2, 2, 3, 2, 'T', 'N'},

        // alpha = 0.5, beta = 0.0
        pdgemm_state{10, 10, 10, 2, 2, 2, 2, 2, 'N', 'N', 0.5, 0.0},
        pdgemm_state{5, 5, 5, 2, 2, 2, 2, 2, 'N', 'N', 0.5, 0.0},
        pdgemm_state{5, 5, 5, 2, 2, 2, 2, 2, 'T', 'N', 0.5, 0.0},
        pdgemm_state{8, 4, 8, 2, 2, 2, 3, 2, 'N', 'N', 0.5, 0.0},
        pdgemm_state{8, 4, 8, 2, 2, 2, 3, 2, 'T', 'N', 0.5, 0.0}
    ));

