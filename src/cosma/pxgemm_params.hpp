// a container class, containing all the parameters of pxgemm
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <cosma/scalapack.hpp>

namespace cosma {
template <typename T>
struct pxgemm_params {
    // ****************************************
    // *       INPUT PARAMETERS BEGIN         *
    // ****************************************
    // *  global dimensions  *
    // ***********************
    // matrix A
    int ma; // rows
    int na; // cols

    // matrix B
    int mb; // rows
    int nb; // cols

    // matrix C
    int mc; // rows
    int nc; // cols

    // ***********************
    // *     block sizes     *
    // ***********************
    // matrix A
    int bma; // rows
    int bna; // cols

    // matrix B
    int bmb; // rows
    int bnb; // cols

    // matrix C
    int bmc; // rows
    int bnc; // cols

    // ***********************
    // *   submatrices ij    *
    // ***********************
    // matrix A
    int ia = 1; // rows
    int ja = 1; // cols

    // matrix B
    int ib = 1; // rows
    int jb = 1; // cols

    // matrix C
    int ic = 1; // rows
    int jc = 1; // cols

    // ***********************
    // *    problem size     *
    // ***********************
    int m;
    int n;
    int k;

    // ***********************
    // *   transpose flags   *
    // ***********************
    char trans_a = 'N';
    char trans_b = 'N';

    // ***********************
    // *    scaling flags    *
    // ***********************
    T alpha = T{1};
    T beta = T{0};

    // ***********************
    // *    leading dims     *
    // ***********************
    int lld_a;
    int lld_b;
    int lld_c;

    // ***********************
    // *      proc grid      *
    // ***********************
    int p_rows; // rows
    int p_cols; // cols
    int P;
    char order = 'R';

    // ***********************
    // *      proc srcs      *
    // ***********************
    // matrix A
    int src_ma = 0; // rows
    int src_na = 0; // cols

    // matrix B
    int src_mb = 0; // rows
    int src_nb = 0; // cols

    // matrix C
    int src_mc = 0; // rows
    int src_nc = 0; // cols

    // ****************************************
    // *         INPUT PARAMETERS END         *
    // ****************************************
    pxgemm_params() = default;

    void initialize(int mm, int nn, int kk,
                    int block_a1, int block_a2,
                    int block_b1, int block_b2,
                    int block_c1, int block_c2,
                    int prows, int pcols,
                    char transa, char transb,
                    T a, T b) {
        m = mm;
        n = nn;
        k = kk;

        // global problem size
        // m, n, k are just sizes that we want to multiply
        // starting from (ia-1, ja-1), (ib-1, jb-1) and (ic-1, jc-1)
        // this makes the global problem size m+ia-1, n+jb-1, k+ja-1
        // use transa instead of trans_a since trans_a is set afterwards
        ma = transpose_if(transa, k, m);
        na = transpose_if(transa, m, k);

        mb = transpose_if(transb, n, k);
        nb = transpose_if(transb, k, n);
        mc = m;
        nc = n;

        // block sizes
        bma = block_a1;
        bna = block_a2;

        bmb = block_b1;
        bnb = block_b2;

        bmc = block_c1;
        bnc = block_c2;

        // submatrices ij
        ia = 1; ja = 1;
        ib = 1; jb = 1;
        ic = 1; jc = 1;

        // transpose flags
        trans_a = std::toupper(transa);
        trans_b = std::toupper(transb);

        // scaling parameters
        alpha = a;
        beta = b;

        // proc grid
        order = 'R';
        p_rows = prows;
        p_cols = pcols;
        P = p_rows * p_cols;

        // leading dims
        lld_a = scalapack::max_leading_dimension(ma, bma, p_rows);
        lld_b = scalapack::max_leading_dimension(mb, bmb, p_rows);
        lld_c = scalapack::max_leading_dimension(mc, bmc, p_rows);

        // proc srcs
        src_ma = 0; src_na = 0;
        src_mb = 0; src_nb = 0;
        src_mc = 0; src_nc = 0;
    }

    pxgemm_params(int m, int n, int k,
                  int bm, int bn, int bk,
                  int prows, int pcols,
                  char transa, char transb,
                  T a, T b) {
        // block sizes
        // blocks BEFORE transposing (if transposed)
        bma = transpose_if(transa, bk, bm);
        bna = transpose_if(transa, bm, bk);

        bmb = transpose_if(transb, bn, bk);
        bnb = transpose_if(transb, bk, bn);

        bmc = bm;
        bnc = bn;

        initialize(m, n, k,
                   bma, bna,
                   bmb, bnb,
                   bmc, bnc,
                   prows, pcols,
                   transa, transb,
                   a, b);
        std::string info;
        if (!valid(info)) {
            std::runtime_error("WRONG PXGEMM PARAMETER: " + info);
        }
    }


    pxgemm_params(int m, int n, int k,
                  int block_a1, int block_a2,
                  int block_b1, int block_b2,
                  int block_c1, int block_c2,
                  int prows, int pcols,
                  char transa, char transb,
                  T a, T b) {
        initialize(m, n, k,
                   block_a1, block_a2,
                   block_b1, block_b2,
                   block_c1, block_c2,
                   prows, pcols,
                   transa, transb,
                   a, b);
        std::string info;
        if (!valid(info)) {
            std::runtime_error("WRONG PXGEMM PARAMETER: " + info);
        }
    }

    pxgemm_params(
        // global sizes
        int ma, int na, // matrix A
        int mb, int nb, // matrix B
        int mc, int nc, // matrix C

        // block sizes
        int bma, int bna, // matrix A
        int bmb, int bnb, // matrix B
        int bmc, int bnc, // matrix C

        // submatrices ij
        int ia, int ja, // matrix A
        int ib, int jb, // matrix B
        int ic, int jc, // matrix C

        // problem size
        int m, int n, int k,

        // transpose flags
        char trans_a, char trans_b,

        // scaling flags
        T alpha, T beta,

        // leading dimensions
        int lld_a, int lld_b, int lld_c,

        // processor grid
        int p_rows, int p_cols,
        char order,

        // processor srcs
        int src_ma, int src_na, // matrix A
        int src_mb, int src_nb, // matrix B
        int src_mc, int src_nc // matrix C
    ) :
        ma(ma), na(na),
        mb(mb), nb(nb),
        mc(mc), nc(nc),

        bma(bma), bna(bna),
        bmb(bmb), bnb(bnb),
        bmc(bmc), bnc(bnc),

        ia(ia), ja(ja),
        ib(ib), jb(jb),
        ic(ic), jc(jc),

        m(m), n(n), k(k),

        trans_a(std::toupper(trans_a)),
        trans_b(std::toupper(trans_b)),

        alpha(alpha), beta(beta),

        lld_a(lld_a), lld_b(lld_b), lld_c(lld_c),

        order(std::toupper(order)),
        p_rows(p_rows), p_cols(p_cols),
        P(p_rows * p_cols),

        src_ma(src_ma), src_na(src_na),
        src_mb(src_mb), src_nb(src_nb),
        src_mc(src_mc), src_nc(src_nc)
    {
        std::string info;
        if (!valid(info)) {
            std::runtime_error("WRONG PXGEMM PARAMETER: " + info);
        }
    }

    int transpose_if(char transpose_flag, int row, int col) {
        bool transposed = transpose_flag != 'N';
        int result = transposed ? row : col;
        return result;
    }

    // if parameters are correct:
    //     returns true is returned and info = "";
    // else:
    //     returns false and info = name of the incorrectly set variable;
    bool valid(std::string& info) {
        info = "";
        // *************************************************
        // check if transpose flags have proper values
        // *************************************************
        std::vector<char> t_flags = {'N', 'T', 'C'};
        if (std::find(t_flags.begin(), t_flags.end(), trans_a) == t_flags.end()) {
            info = "trans_a = " + std::to_string(trans_a);
            return false;
        }
        if (std::find(t_flags.begin(), t_flags.end(), trans_b) == t_flags.end()) {
            info = "trans_b = " + std::to_string(trans_b);
            return false;
        }
        if (order != 'R' && order != 'C') {
            info = "oder = " + std::to_string(order);
            return false;
        }

        // *************************************************
        // check if the following values are all positive
        // *************************************************
        std::vector<int> positive = {
             ma, na, mb, nb, mc, nc,
             bma, bna, bmb, bnb, bmc, bnc,
             m, n, k,
             lld_a, lld_b, lld_c,
             p_rows, p_cols, P,
        };
        std::vector<std::string> positive_labels = {
             "ma", "na", "mb", "nb", "mc", "nc",
             "bma", "bna", "bmb", "bnb", "bmc", "bnc",
             "m", "n", "k",
             "lld_a", "lld_b", "lld_c",
             "p_rows", "p_cols", "P"
        };
        for (int i = 0; i < positive.size(); ++i) {
            if (positive[i] < 0) {
                info = positive_labels[i] + " = " + std::to_string(positive[i]);
                return false;
            }
        }

        // *************************************************
        // check if submatrix start index  
        // is inside the global matrix
        // *************************************************
        // matrix A
        if (ia < 1 || ia > ma) {
            info = "ia = " + std::to_string(ia);
            return false;
        }
        if (ja < 1 || ja > na) {
            info = "ja = " + std::to_string(ja);
            return false;
        }

        // matrix B
        if (ib < 1 || ib > mb) {
            info = "ib = " + std::to_string(ib);
            return false;
        }
        if (jb < 1 || jb > nb) {
            info = "jb = " + std::to_string(jb);
            return false;
        }

        // matrix C
        if (ic < 1 || ic > mc) {
            info = "ic = " + std::to_string(ic);
            return false;
        }
        if (jc < 1 || jc > nc) {
            info = "jc = " + std::to_string(jc);
            return false;
        }

        // *************************************************
        // check if submatrix end index
        // is inside the global matrix
        // *************************************************
        // matrix A
        int ma_sub = transpose_if(trans_a, k, m);
        // guaranteed to be >= ia 
        // (since we previously checked ma_sub >= 1 and ia >= 1)
        int ma_sub_end = ia - 1 + ma_sub;
        if (ma_sub_end >= ma) {
            info = "ia - 1 + (m or k) = " + std::to_string(ma_sub_end);
            return false;
        }
        int na_sub = transpose_if(trans_a, m, k);
        // guaranteed to be >= ja 
        // (since we previously checked na_sub >= 1 and ja >= 1)
        int na_sub_end = ja - 1 + na_sub;
        if (na_sub_end >= na) {
            info = "ja - 1 + (k or m) = " + std::to_string(na_sub_end);
            return false;
        }

        // matrix B
        int mb_sub = transpose_if(trans_b, n, k);
        // guaranteed to be >= ib 
        // (since we previously checked mb_sub >= 1 and ib >= 1)
        int mb_sub_end = ib - 1 + mb_sub;
        if (mb_sub_end >= mb) {
            info = "ib - 1 + (k or n) = " + std::to_string(mb_sub_end);
            return false;
        }
        int nb_sub = transpose_if(trans_b, k, n);
        // guaranteed to be >= jb 
        // (since we previously checked nb_sub >= 1 and jb >= 1)
        int nb_sub_end = jb - 1 + nb_sub;
        if (nb_sub_end >= nb) {
            info = "jb - 1 + (n or k) = " + std::to_string(nb_sub_end);
            return false;
        }

        // matrix C
        int mc_sub = m;
        // guaranteed to be >= ic 
        // (since we previously checked mc_sub >= 1 and ic >= 1)
        int mc_sub_end = ic - 1 + mc_sub;
        if (mc_sub_end >= mc) {
            info = "ic - 1 + m = " + std::to_string(mc_sub_end);
            return false;
        }
        int nc_sub = n;
        // guaranteed to be >= jc 
        // (since we previously checked nc_sub >= 1 and jc >= 1)
        int nc_sub_end = jc - 1 + nc_sub;
        if (nc_sub_end >= nc) {
            info = "jc - 1 + n = " + std::to_string(nc_sub_end);
            return false;
        }

        // *************************************************
        // check if row/col src elements are within the 
        // global dimensions of matrices
        // *************************************************
        // matrix A
        if (src_ma < 0 || src_ma >= ma) {
            info = "src_ma = " + std::to_string(src_ma);
            return false;
        }
        if (src_na < 0 || src_na >= na) {
            info = "src_na = " + std::to_string(src_na);
            return false;
        }

        // matrix B
        if (src_mb < 0 || src_mb >= mb) {
            info = "src_mb = " + std::to_string(src_mb);
            return false;
        }
        if (src_nb < 0 || src_nb >= nb) {
            info = "src_nb = " + std::to_string(src_nb);
            return false;
        }

        // matrix C
        if (src_mc < 0 || src_mc >= mc) {
            info = "src_mc = " + std::to_string(src_mc);
            return false;
        }
        if (src_nc < 0 || src_nc >= nc) {
            info = "src_nc = " + std::to_string(src_nc);
            return false;
        }

        // *************************************************
        // check leading dimensions
        // *************************************************
        int min_lld_a = scalapack::min_leading_dimension(ma, bma, p_rows);
        int min_lld_b = scalapack::min_leading_dimension(mb, bmb, p_rows);
        int min_lld_c = scalapack::min_leading_dimension(mc, bmc, p_rows);

        if (lld_a < min_lld_a) {
            info = "lld_a = " + std::to_string(min_lld_a);
            return false;
        }
        if (lld_b < min_lld_b) {
            info = "lld_b = " + std::to_string(min_lld_b);
            return false;
        }
        if (lld_c < min_lld_c) {
            info = "lld_c = " + std::to_string(min_lld_c);
            return false;
        }

        return true;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const pxgemm_params &obj) {
        os << "=============================" << std::endl;
        os << "      GLOBAL MAT. SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "A = " << obj.ma << " x " << obj.na << std::endl;
        os << "B = " << obj.mb << " x " << obj.nb << std::endl;
        os << "C = " << obj.mc << " x " << obj.nc << std::endl;
        os << "=============================" << std::endl;
        os << "        SUBMATRICES" << std::endl;
        os << "=============================" << std::endl;
        os << "(ia, ja) = (" << obj.ia << ", " << obj.ja << ")" << std::endl;
        os << "(ib, jb) = (" << obj.ib << ", " << obj.jb << ")" << std::endl;
        os << "(ic, jc) = (" << obj.ic << ", " << obj.jc << ")" << std::endl;
        os << "=============================" << std::endl;
        os << "      SUBMATRIX SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "m = " << obj.m << std::endl;
        os << "n = " << obj.n << std::endl;
        os << "k = " << obj.k << std::endl;
        os << "=============================" << std::endl;
        os << "      ADDITIONAL OPTIONS" << std::endl;
        os << "=============================" << std::endl;
        os << "alpha = " << obj.alpha << std::endl;
        os << "beta = " << obj.beta << std::endl;
        os << "trans_a = " << obj.trans_a << std::endl;
        os << "trans_b = " << obj.trans_b << std::endl;
        os << "=============================" << std::endl;
        os << "         PROC GRID" << std::endl;
        os << "=============================" << std::endl;
        os << "grid = " << obj.p_rows << " x " << obj.p_cols << std::endl;
        os << "grid order = " << obj.order << std::endl;
        os << "=============================" << std::endl;
        os << "         PROC SRCS" << std::endl;
        os << "=============================" << std::endl;
        os << "P_SRC(A) = (" << obj.src_ma << ", " << obj.src_na << ")" << std::endl;
        os << "P_SRC(B) = (" << obj.src_mb << ", " << obj.src_nb << ")" << std::endl;
        os << "P_SRC(C) = (" << obj.src_mc << ", " << obj.src_nc << ")" << std::endl;
        os << "=============================" << std::endl;
        os << "          BLOCK SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "Blocks(A) = (" << obj.bma << ", " << obj.bna << ")" << std::endl;
        os << "Blocks(B) = (" << obj.bmb << ", " << obj.bnb << ")" << std::endl;
        os << "Blocks(C) = (" << obj.bmc << ", " << obj.bnc << ")" << std::endl;
        os << "=============================" << std::endl;
        os << "          LEADING DIMS" << std::endl;
        os << "=============================" << std::endl;
        os << "lld_a = " << obj.lld_a << std::endl;
        os << "lld_b = " << obj.lld_b << std::endl;
        os << "lld_c = " << obj.lld_c << std::endl;
        os << "=============================" << std::endl;
        return os;
    }
};
}
