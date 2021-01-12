#include <cmath>
#include <tuple>

namespace costa {
std::pair<int, int> inverse_cantor_pairing(int z) {
    int w = (int)std::floor((std::sqrt(8 * z + 1) - 1) / 2);
    int t = (w * w + w) / 2;
    int y = z - t;
    int x = w - y;
    return {x, y};
}

// maps (N, N) -> N
int cantor_pairing(const int i, const int j) {
    int sum = i + j;
    return (sum * (sum + 1)) / 2 + j;
}
} // namespace costa
