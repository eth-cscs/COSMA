#include "../gtest.h"
#include <mapper.hpp>

TEST(mapper, bdb) {
    auto m = 8u;
    auto n = 4u;
    auto k = 2u;
    auto n_steps = 3u;
    std::string patt = "bdb";
    std::vector<int> divPatt = {2, 1, 1, 2, 1, 1, 1, 2, 1};
    auto P = 4u;

    Mapper A('A', m, k, P, n_steps, 0, 2, patt.cbegin(), divPatt.cbegin(), 0);
    Mapper B('B', k, n, P, n_steps, 2, 1, patt.cbegin(), divPatt.cbegin(), 0);
    Mapper C('C', m, n, P, n_steps, 0, 1, patt.cbegin(), divPatt.cbegin(), 0);

    // test initial sizes for all ranks
    std::vector<int> A_initial_size_target = {4, 4, 4, 4};
    std::vector<int> B_initial_size_target = {2, 2, 2, 2};
    std::vector<int> C_initial_size_target = {8, 8, 8, 8};

    for (auto i = 0u; i < P; ++i) {
        EXPECT_EQ(A.initial_size(i), A_initial_size_target[i]);
        EXPECT_EQ(B.initial_size(i), B_initial_size_target[i]);
        EXPECT_EQ(C.initial_size(i), C_initial_size_target[i]);
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

