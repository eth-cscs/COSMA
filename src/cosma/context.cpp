#include <cosma/context.hpp>

#include <complex>

namespace cosma {

// constructor
template <typename Scalar>
context<Scalar>::context() {
#ifdef COSMA_HAVE_GPU
    gpu_ctx = gpu::make_context<Scalar>();
#endif
}

// constructor
template <typename Scalar>
context<Scalar>::context(int streams, int tile_m, int tile_n, int tile_k) {
#ifdef COSMA_HAVE_GPU
    gpu_ctx = gpu::make_context<Scalar>(streams, tile_m, tile_n, tile_k);
#else
    std::cout << "Ignoring parameters in make_context. These parameters only "
                 "used in the CPU version."
              << std::endl;
#endif
}

using zfloat = std::complex<float>;
using zdouble = std::complex<double>;

template class context<float>;
template class context<double>;
template class context<zfloat>;
template class context<zdouble>;

} // namespace cosma
