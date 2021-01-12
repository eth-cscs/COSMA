// a container class, containing all the parameters of pxgemr2d
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <costa/scalapack.hpp>

namespace costa {
template <typename T>
struct pxgemr2d_params {
    // ****************************************
    // *       INPUT PARAMETERS BEGIN         *
    // ****************************************
    // *  global dimensions  *
    // ***********************
    // matrix A
    int ma; // rows
    int na; // cols

    // matrix C
    int mc; // rows
    int nc; // cols

    // ***********************
    // *     block sizes     *
    // ***********************
    // matrix A
    int bma; // rows
    int bna; // cols

    // matrix C
    int bmc; // rows
    int bnc; // cols

    // ***********************
    // *   submatrices ij    *
    // ***********************
    // matrix A
    int ia = 1; // rows
    int ja = 1; // cols

    // matrix C
    int ic = 1; // rows
    int jc = 1; // cols

    // ***********************
    // *    problem size     *
    // ***********************
    int m;
    int n;

    // ***********************
    // *    leading dims     *
    // ***********************
    int lld_a;
    int lld_c;

    // ***********************
    // *      proc grid      *
    // ***********************
    int p_rows_a; // rows
    int p_cols_a; // cols

    int p_rows_c; // rows
    int p_cols_c; // cols

    int P;

    char order_a = 'R';
    char order_c = 'R';

    // ***********************
    // *      proc srcs      *
    // ***********************
    // matrix A
    int src_ma = 0; // rows
    int src_na = 0; // cols

    // matrix C
    int src_mc = 0; // rows
    int src_nc = 0; // cols

    // ****************************************
    // *         INPUT PARAMETERS END         *
    // ****************************************
    pxgemr2d_params() = default;

    void initialize(int mm, int nn,
                    int block_a1, int block_a2,
                    int block_c1, int block_c2,
                    int prows_a, int pcols_a,
                    int prows_c, int pcols_c
                    ) {
        m = mm;
        n = nn;

        // global problem size
        // m, n are just the global sizes
        // starting from (ia-1, ja-1) and (ic-1, jc-1)
        // this makes the global problem size m+ia-1, k+ja-1
        // a is to be transposed
        ma = m;
        na = n;
        mc = m;
        nc = n;

        // block sizes
        bma = block_a1;
        bna = block_a2;

        bmc = block_c1;
        bnc = block_c2;

        // submatrices ij
        ia = 1; ja = 1;
        ic = 1; jc = 1;

        // proc grid
        order_a = 'R';
        order_c = 'R';
        p_rows_a = prows_a;
        p_cols_a = pcols_a;
        p_rows_c = prows_c;
        p_cols_c = pcols_c;
        P = std::max(p_rows_a * p_cols_a, p_rows_c * p_cols_c);

        // leading dims
        lld_a = scalapack::max_leading_dimension(ma, bma, p_rows_a);
        lld_c = scalapack::max_leading_dimension(mc, bmc, p_rows_c);

        // proc srcs
        src_ma = 0; src_na = 0;
        src_mc = 0; src_nc = 0;
    }

    pxgemr2d_params(int m, int n,
                  int bm, int bn,
                  int prows_a, int pcols_a,
                  int prows_c, int pcols_c
                  ) {
        // block sizes
        // blocks BEFORE transposing (if transposed)
        bma = bm;
        bna = bn;

        bmc = bm;
        bnc = bn;

        initialize(m, n,
                   bma, bna,
                   bmc, bnc,
                   prows_a, pcols_a,
                   prows_c, pcols_c
                   );
        std::string info;
        if (!valid(info)) {
            std::runtime_error("WRONG PXGEMR2D PARAMETER: " + info);
        }
    }


    pxgemr2d_params(int m, int n,
                  int block_a1, int block_a2,
                  int block_c1, int block_c2,
                  int prows_a, int pcols_a,
                  int prows_c, int pcols_c
                  ) {
        initialize(m, n,
                   block_a1, block_a2,
                   block_c1, block_c2,
                   prows_a, pcols_a,
                   prows_c, pcols_c
                   );
        std::string info;
        if (!valid(info)) {
            std::runtime_error("WRONG PXGEMR2D PARAMETER: " + info);
        }
    }

    pxgemr2d_params(
        // global sizes
        int ma, int na, // matrix A
        int mc, int nc, // matrix C

        // block sizes
        int bma, int bna, // matrix A
        int bmc, int bnc, // matrix C

        // submatrices ij
        int ia, int ja, // matrix A
        int ic, int jc, // matrix C

        // problem size
        int m, int n,

        // leading dimensions
        int lld_a, int lld_c,

        // processor grid
        int p_rows_a, int p_cols_a,
        int p_rows_c, int p_cols_c,

        // processor grid order
        char order_a,
        char order_c,

        // processor srcs
        int src_ma, int src_na, // matrix A
        int src_mc, int src_nc // matrix C
    ) :
        ma(ma), na(na),
        mc(mc), nc(nc),

        bma(bma), bna(bna),
        bmc(bmc), bnc(bnc),

        ia(ia), ja(ja),
        ic(ic), jc(jc),

        m(m), n(n),

        lld_a(lld_a), lld_c(lld_c),

        order_a(std::toupper(order_a)),
        order_c(std::toupper(order_c)),
        p_rows_a(p_rows_a), p_cols_c(p_cols_c),
        P(std::max(p_rows_a * p_cols_a, p_rows_c * p_cols_c)),

        src_ma(src_ma), src_na(src_na),
        src_mc(src_mc), src_nc(src_nc)
    {
        std::string info;
        if (!valid(info)) {
            std::runtime_error("WRONG PXGEMR2D PARAMETER: " + info);
        }
    }

    // if parameters are correct:
    //     returns true is returned and info = "";
    // else:
    //     returns false and info = name of the incorrectly set variable;
    bool valid(std::string& info) {
        info = "";
        if (order_a != 'R' && order_a != 'C') {
            info = "oder_a = " + std::to_string(order_a);
            return false;
        }
        if (order_c != 'R' && order_c != 'C') {
            info = "oder_c = " + std::to_string(order_c);
            return false;
        }

        // *************************************************
        // check if the following values are all positive
        // *************************************************
        std::vector<int> positive = {
             ma, na, mc, nc,
             bma, bna, bmc, bnc,
             m, n,
             lld_a, lld_c,
             p_rows_a, p_cols_a,
             p_rows_c, p_cols_c,
             P
        };
        std::vector<std::string> positive_labels = {
             "ma", "na", "mc", "nc",
             "bma", "bna", "bmc", "bnc",
             "m", "n",
             "lld_a", "lld_c",
             "p_rows_a", "p_cols_a",
             "p_rows_c", "p_cols_c",
             "P"
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
        int ma_sub = n;
        // guaranteed to be >= ia
        // (since we previously checked ma_sub >= 1 and ia >= 1)
        int ma_sub_end = ia - 1 + ma_sub;
        if (ma_sub_end >= ma) {
            info = "ia - 1 + (m or k) = " + std::to_string(ma_sub_end);
            return false;
        }
        int na_sub = m;
        // guaranteed to be >= ja
        // (since we previously checked na_sub >= 1 and ja >= 1)
        int na_sub_end = ja - 1 + na_sub;
        if (na_sub_end >= na) {
            info = "ja - 1 + (k or m) = " + std::to_string(na_sub_end);
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
        int min_lld_a = scalapack::min_leading_dimension(ma, bma, p_rows_a);
        int min_lld_c = scalapack::min_leading_dimension(mc, bmc, p_rows_c);

        if (lld_a < min_lld_a) {
            info = "lld_a = " + std::to_string(min_lld_a);
            return false;
        }
        if (lld_c < min_lld_c) {
            info = "lld_c = " + std::to_string(min_lld_c);
            return false;
        }

        return true;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const pxgemr2d_params &obj) {
        os << "=============================" << std::endl;
        os << "      GLOBAL MAT. SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "A = " << obj.ma << " x " << obj.na << std::endl;
        os << "C = " << obj.mc << " x " << obj.nc << std::endl;
        os << "=============================" << std::endl;
        os << "        SUBMATRICES" << std::endl;
        os << "=============================" << std::endl;
        os << "(ia, ja) = (" << obj.ia << ", " << obj.ja << ")" << std::endl;
        os << "(ic, jc) = (" << obj.ic << ", " << obj.jc << ")" << std::endl;
        os << "=============================" << std::endl;
        os << "      SUBMATRIX SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "m = " << obj.m << std::endl;
        os << "n = " << obj.n << std::endl;
        os << "=============================" << std::endl;
        os << "         PROC GRID" << std::endl;
        os << "=============================" << std::endl;
        os << "grid_a = " << obj.p_rows_a << " x " << obj.p_cols_a << std::endl;
        os << "grid_c = " << obj.p_rows_c << " x " << obj.p_cols_c << std::endl;
        os << "grid order_a = " << obj.order_a << std::endl;
        os << "grid order_c = " << obj.order_c << std::endl;
        os << "=============================" << std::endl;
        os << "         PROC SRCS" << std::endl;
        os << "=============================" << std::endl;
        os << "P_SRC(A) = (" << obj.src_ma << ", " << obj.src_na << ")" << std::endl;
        os << "P_SRC(C) = (" << obj.src_mc << ", " << obj.src_nc << ")" << std::endl;
        os << "=============================" << std::endl;
        os << "          BLOCK SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "Blocks(A) = (" << obj.bma << ", " << obj.bna << ")" << std::endl;
        os << "Blocks(C) = (" << obj.bmc << ", " << obj.bnc << ")" << std::endl;
        os << "=============================" << std::endl;
        os << "          LEADING DIMS" << std::endl;
        os << "=============================" << std::endl;
        os << "lld_a = " << obj.lld_a << std::endl;
        os << "lld_c = " << obj.lld_c << std::endl;
        os << "=============================" << std::endl;
        return os;
    }
};
}
