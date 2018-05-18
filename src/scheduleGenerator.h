#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <assert.h>

#define MAX_SIZE 2147483640 

namespace spartition {

    enum dim {dimM, dimN, dimK};

    enum splitType {BFS, DFS};

    enum schedType {S2D, S3D};

    enum DivisionStrategy { oneStep, recursive };

    struct SingleStep {
        dim Dim;
        splitType SplitType;
        int SplitSize;
    };

    struct Schedule {
        std::vector<SingleStep> divisions;
        int tileSizeM, tileSizeN, tileSizeK;
        int numTilesM, numTilesN, numTilesK;
    };

    struct ProblemParameters {
        int m, n, k; //dimensions
        long long S; //size of local memory
        int N; //number of cores per node
        int P; //number of processes (ranks)
        schedType schedule; //generates 2d or 3d schedule (parallelization in k dimension)
        DivisionStrategy divStrat; //e.g., if schedule should look like 16 or 2, 2, 2, 2
    };

    std::vector<int> Factorize(int);

    Schedule GenerateSchedule(ProblemParameters params);
}
