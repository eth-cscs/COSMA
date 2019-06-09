#include <cosma/mapper.hpp>

#include <gtest.h>

using namespace cosma;

TEST(strategy, spartition) {
    int m = 17408;
    int n = 17408;
    int k = 3735552;
    long long memory_limit = 80000000; // #elements, per node, corresponding to 50GB
    int nodes = 64;
    int ranks_per_node = 36;
    int P = nodes * ranks_per_node;
    // memory_limit /= ranks_per_node;

    Strategy strategy(m, n, k, P, memory_limit);

    std::cout << "Strategy = " << strategy << std::endl;
    std::cout << "n seq steps = " << strategy.n_sequential_steps << std::endl;
    EXPECT_TRUE(strategy.n_sequential_steps > 0);
}

TEST(strategy, nested_sequential_parallel) {
    int m = 30000;
    int n = m;
    int k = m;
    long long memory_limit = 80000000; // #elements, per node, corresponding to 50GB
    int nodes = 10;
    int ranks_per_node = 36;
    int P = nodes * ranks_per_node;
    // memory_limit /= ranks_per_node;

    Strategy strategy(m, n, k, P, memory_limit);

    std::cout << "Strategy = " << strategy << std::endl;
    std::cout << "n seq steps = " << strategy.n_sequential_steps << std::endl;
    // EXPECT_TRUE(strategy.n_seq_steps > 0);
}

TEST(mapper, bdb) {
    auto m = 8u;
    auto n = 4u;
    auto k = 2u;
    std::string types = "psp";
    std::vector<int> divisors = {2, 2, 2};
    std::string dims = "mmn";
    auto P = 4u;

    Strategy strategy(m, n, k, P, divisors, dims, types);

    // the last element is rank = 0, but regardless of this parameter
    // the mapper can compute the buffers sizes or the mapper for any rank
    Mapper A('A', m, k, P, strategy, 0);
    Mapper B('B', k, n, P, strategy, 0);
    Mapper C('C', m, n, P, strategy, 0);

    // test initial sizes for all ranks
    std::vector<int> A_initial_size_target = {4, 4, 4, 4};
    std::vector<int> B_initial_size_target = {2, 2, 2, 2};
    std::vector<int> C_initial_size_target = {8, 8, 8, 8};

    for (auto i = 0u; i < P; ++i) {
        EXPECT_EQ(A.initial_size(i), A_initial_size_target[i]);
        EXPECT_EQ(B.initial_size(i), B_initial_size_target[i]);
        EXPECT_EQ(C.initial_size(i), C_initial_size_target[i]);
    }
    // check if the precomputed local->global mapper for 
    // local elements on the current rank match
    // the results from the on-demand computed local->global
    // mapper for any element and any rank
    // we only check if this holds for rank 0
    for (auto i = 0u; i < A_initial_size_target[0]; ++i) {
        int gi, gj;
        std::tie(gi, gj) = A.global_coordinates(i, 0);
        int gi_local, gj_local;
        std::tie(gi_local, gj_local) = A.global_coordinates(i);
        EXPECT_EQ(gi, gi_local);
        EXPECT_EQ(gj, gj_local);
    }

    for (auto i = 0u; i < B_initial_size_target[0]; ++i) {
        int gi, gj;
        std::tie(gi, gj) = B.global_coordinates(i, 0);
        int gi_local, gj_local;
        std::tie(gi_local, gj_local) = B.global_coordinates(i);
        EXPECT_EQ(gi, gi_local);
        EXPECT_EQ(gj, gj_local);
    }

    for (auto i = 0u; i < C_initial_size_target[0]; ++i) {
        int gi, gj;
        std::tie(gi, gj) = C.global_coordinates(i, 0);
        int gi_local, gj_local;
        std::tie(gi_local, gj_local) = C.global_coordinates(i);
        EXPECT_EQ(gi, gi_local);
        EXPECT_EQ(gj, gj_local);
    }

    // test rank_to_range map which specified for each rank, the list of Interval2D it owns
    std::vector<std::vector<Interval2D>> A_rank_to_range_target = 
            {{Interval2D(0, 1, 0, 0), Interval2D(2, 3, 0, 0)},
             {Interval2D(0, 1, 1, 1), Interval2D(2, 3, 1, 1)},
             {Interval2D(4, 5, 0, 0), Interval2D(6, 7, 0, 0)},
             {Interval2D(4, 5, 1, 1), Interval2D(6, 7, 1, 1)}};

    std::vector<Interval2D> B_rank_to_range_target = 
            {Interval2D(0, 1, 0, 0),
             Interval2D(0, 1, 2, 2),
             Interval2D(0, 1, 1, 1),
             Interval2D(0, 1, 3, 3)};

    std::vector<std::vector<Interval2D>> C_rank_to_range_target = 
            {{Interval2D(0, 1, 0, 1), Interval2D(2, 3, 0, 1)},
             {Interval2D(0, 1, 2, 3), Interval2D(2, 3, 2, 3)},
             {Interval2D(4, 5, 0, 1), Interval2D(6, 7, 0, 1)},
             {Interval2D(4, 5, 2, 3), Interval2D(6, 7, 2, 3)}};

    auto A_rank_to_range = A.complete_layout();
    auto B_rank_to_range = B.complete_layout();
    auto C_rank_to_range = C.complete_layout();

    for (auto i = 0u; i < P; ++i) {
        EXPECT_EQ(A_rank_to_range[i].size(), 2);
        EXPECT_EQ(B_rank_to_range[i].size(), 1);
        EXPECT_EQ(C_rank_to_range[i].size(), 2);

        for (auto range = 0u; range < A_rank_to_range[i].size(); ++range) {
            EXPECT_EQ(A_rank_to_range[i][range], A_rank_to_range_target[i][range]);
        }

        EXPECT_EQ(B_rank_to_range[i][0], B_rank_to_range_target[i]);

        for (auto range = 0u; range < C_rank_to_range[i].size(); ++range) {
            EXPECT_EQ(C_rank_to_range[i][range], C_rank_to_range_target[i][range]);
        }
    }

    // test the mapping global<->local
    std::vector<std::pair<int, int>> A_global_coord = {{0, 0}, {1, 0}, {2, 0}, {3, 0},
                                                       {0, 1}, {1, 1}, {2, 1}, {3, 1},
                                                       {4, 0}, {5, 0}, {6, 0}, {7, 0},
                                                       {4, 1}, {5, 1}, {6, 1}, {7, 1}};

    std::vector<std::pair<int, int>> B_global_coord = {{0, 0}, {1, 0},
                                                       {0, 2}, {1, 2},
                                                       {0, 1}, {1, 1},
                                                       {0, 3}, {1, 3}};

    auto i = 0u;
    for(auto rank = 0u; rank < P; ++rank) {
        for(auto locIdx = 0u; locIdx < A_initial_size_target[rank]; ++locIdx) {
            int gi, gj;
            std::tie(gi, gj) = A.global_coordinates(locIdx, rank);
            EXPECT_EQ(A_global_coord[i].first, gi);
            EXPECT_EQ(A_global_coord[i].second, gj);

            int l, r;
            std::tie(l, r) = A.local_coordinates(gi, gj);
            EXPECT_EQ(l, locIdx);
            EXPECT_EQ(r, rank);
            ++i;
        }
    }

    i = 0u;
    for(auto rank = 0u; rank < P; ++rank) {
        for(auto locIdx = 0u; locIdx < B_initial_size_target[rank]; ++locIdx) {
            int gi, gj;
            std::tie(gi, gj) = B.global_coordinates(locIdx, rank);
            EXPECT_EQ(B_global_coord[i].first, gi);
            EXPECT_EQ(B_global_coord[i].second, gj);

            int l, r;
            std::tie(l, r) = B.local_coordinates(gi, gj);
            EXPECT_EQ(l, locIdx);
            EXPECT_EQ(r, rank);
            ++i;
        }
    }
}

