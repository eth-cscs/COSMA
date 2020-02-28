/*
Simulates the algorithm (without actually computing the matrix multiplication)
 * in order to get the total volume of the communication, the maximum volume of computation
 * done in a single branch and the maximum required buffer size that the algorithm requires.
 */
#include "../utils/parse_strategy.hpp"
#include <cosma/statistics.hpp>

#include <iostream>

using namespace cosma;

int main( int argc, char **argv ) {
    const Strategy& strategy = parse_strategy(argc, argv);

    std::cout << strategy << std::endl;

    int n_rep = 1;
    multiply(strategy, n_rep);

    return 0;
}
