#pragma once

#include <complex>
#include <random>

namespace cosma {
/*
 * generators a random int, float, double, 
 * complex<int>, complex<float>, complex<double>
 */
template <typename Scalar>
struct random_generator {
  static inline Scalar sample();
};

template <>
inline int random_generator<int>::sample() {
    static std::random_device dev;                        // seed
    static std::mt19937 rng(dev());                       // generator
    static std::uniform_int_distribution<int> dist(10); // distribution

    return dist(rng);
}

template <>
inline double random_generator<double>::sample() {
    static std::random_device dev;                        // seed
    static std::mt19937 rng(dev());                       // generator
    static std::uniform_real_distribution<double> dist(1.0); // distribution

    return dist(rng);
}

template <>
inline float random_generator<float>::sample() {
    static std::random_device dev;                        // seed
    static std::mt19937 rng(dev());                       // generator
    static std::uniform_real_distribution<float> dist(1.0f); // distribution

    return dist(rng);
}

template <>
inline std::complex<int> random_generator<std::complex<int>>::sample() {
    static std::random_device dev;                        // seed
    static std::mt19937 rng(dev());                       // generator
    static std::uniform_int_distribution<int> dist(10); // distribution
    return {dist(rng), dist(rng)};
}

template <>
inline std::complex<float> random_generator<std::complex<float>>::sample() {
    static std::random_device dev;                        // seed
    static std::mt19937 rng(dev());                       // generator
    static std::uniform_real_distribution<float> dist(1.0f); // distribution
    return {dist(rng), dist(rng)};
}

template <>
inline std::complex<double> random_generator<std::complex<double>>::sample() {
    static std::random_device dev;                        // seed
    static std::mt19937 rng(dev());                       // generator
    static std::uniform_real_distribution<double> dist(1.0); // distribution
    return {dist(rng), dist(rng)};
}
} // end namespace cosma
