#include <cosma/strategy.hpp>

int main(int argc, char **argv) {
    int m = 10000;
    int n = m;
    int k = m;
    int nodes = 10;
    int ranks_per_node = 2;
    int P = nodes * ranks_per_node;

    cosma::Strategy strategy(m, n, k, P);

    std::cout << "Strategy = " << strategy << std::endl;
    return 0;
}
