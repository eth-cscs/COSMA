#ifdef COSMA_WITH_SCALAPACK

// pdgemm wrapper
namespace cosma {
// alpha ignored at the moment
template <typename T>
void pgemm(const char trans_a, const char trans_b, const int m, const int n, const int k,
           const T alpha, const T* a, const int ia, const int ja, const int* desca,
           const T* b, const int ib, const int jb, const int* descb, const T beta,
           T* c, const int ic, const int jc, const int* descc);
}
#endif
