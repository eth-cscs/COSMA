/* Simulates the algorithm (without actually computing the matrix multiplication)
 * in order to get the total volume of the communication, the maximum volume of computation
 * done in a single branch and the maximum required buffer size that the algorithm requires.
 */
#include <iostream>
#include <statistics.hpp>

int main( int argc, char **argv ) {
    Strategy strategy(argc, argv);

    std::cout << strategy << std::endl;

    multiply(strategy);

    return 0;
}
