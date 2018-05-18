#include <strategy.hpp>

int main( int argc, char **argv ) {
    int m = 64000;
    int n = 64000;
    int k = 64000;
    int P = 2304;

    Strategy strategy(m, n, k, P);

    std::cout << strategy << std::endl;

    return 0;
}

