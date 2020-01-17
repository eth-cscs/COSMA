// a container class, containing all the parameters of pxgemm
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
        ma = transpose_if(trans_a, m, k);
        na = transpose_if(trans_a, k, m);
        mb = transpose_if(trans_b, k, n);
        nb = transpose_if(trans_b, n, k);
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
        trans_a = transa; 
        trans_b = transb;

        // scaling parameters
        alpha = a;
        beta = b;

        // leading dims
        int max_n_rows_a = ((ma + bma - 1) / bma) * bma;
        int max_n_rows_b = ((mb + bma - 1) / bmb) * bmb;
        int max_n_rows_c = ((mc + bmc - 1) / bmc) * bmc;

        lld_a = std::max(1, max_n_rows_a);
        lld_b = std::max(1, max_n_rows_b);
        lld_c = std::max(1, max_n_rows_c);

        // proc grid
        order = 'R';
        p_rows = prows;
        p_cols = pcols;
        P = p_rows * p_cols;

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
        bma = transpose_if(trans_a, bk, bm);
        bna = transpose_if(trans_a, bm, bk);

        bmb = transpose_if(trans_a, bn, bk);
        bnb = transpose_if(trans_a, bk, bn);

        bmc = transpose_if(trans_a, bn, bm);
        bnc = transpose_if(trans_a, bm, bn);

        initialize(m, n, k,
                   bma, bna,
                   bmb, bnb,
                   bmc, bnc,
                   prows, pcols,
                   transa, transb,
                   a, b);
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

        trans_a(trans_a), trans_b(trans_b),

        alpha(alpha), beta(beta),

        lld_a(lld_a), lld_b(lld_b), lld_c(lld_c),

        order(order),
        p_rows(p_rows), p_cols(p_cols),
        P(p_rows * p_cols),

        src_ma(src_ma), src_na(src_na),
        src_mb(src_mb), src_nb(src_nb),
        src_mc(src_mc), src_nc(src_nc)
    {}

    int transpose_if(char transpose_flag, int row, int col) {
        return transpose_flag == 'N' ? row : col;
    }

    // TODO: checks if all the parameters make sense
    bool check_correctness() {
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
        os << "P_SRC(A) = (" << obj.src_ma << ", " << obj.src_na << std::endl;
        os << "P_SRC(B) = (" << obj.src_mb << ", " << obj.src_nb << std::endl;
        os << "P_SRC(C) = (" << obj.src_mc << ", " << obj.src_nc << std::endl;
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
