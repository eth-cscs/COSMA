#include "matrix.hpp"

/* Simulates the algorithm (without actually computing the matrix multiplication)
   and outputs the following information:
       * total volume of the communication 
       * maximum volume of computation done in a single branch
       * maximum buffer size that the algorithm requires
       * size of matrix (m, n, k) in the base case with the maximum computational volume
 */

CarmaMatrix* matrixA;
CarmaMatrix* matrixB;
CarmaMatrix* matrixC;

long long total_communication = 0;
long long max_buffer_size = 0;
long long max_total_computation = 0;
int local_m = 0;
int local_n = 0;
int local_k = 0;

void multiply(const Strategy& strategy);

void multiply(Interval& m, Interval& n, Interval& k, Interval& P, int step,
              const Strategy& strategy, double beta, int rank);

void local_multiply(int m, int n, int k, double beta);

void DFS(Interval& m, Interval& n, Interval& k, Interval& P, int step,
         const Strategy& strategy, double beta, int rank);

void BFS(Interval& m, Interval& n, Interval& k, Interval& P, int step,
         const Strategy& strategy, double beta, int rank);
/*
    Compute C = A * B
    m = #rows of A,C
    n = #columns of B,C
    k = #columns of A, #rows of B
    P = #processors involved
    r = #recursive steps
    patt = array of 'B' and 'D' indicating BFS or DFS steps; length r
    divPatt = array of how much each of m,n,k are divided by at each recursive step; length 3*r

    Assumption: we assume that at each step only 1 dimension is split
 */

// this is just a wrapper to initialize and call the recursive function
void multiply(const Strategy& strategy) {
    Interval mi = Interval(0, strategy.m-1);
    Interval ni = Interval(0, strategy.n-1);
    Interval ki = Interval(0, strategy.k-1);
    Interval Pi = Interval(0, strategy.P-1);

    //Declare A,B and C CARMA matrices objects
    matrixA = new CarmaMatrix('A', strategy, 0);
    matrixB = new CarmaMatrix('B', strategy, 0);
    matrixC = new CarmaMatrix('C', strategy, 0);

    // simulate the algorithm for each rank
    for (int rank = 0; rank < Pi.length(); ++rank) {
        multiply(mi, ni, ki, Pi, 0, strategy, 0.0, rank);
    }

    free(matrixA);
    free(matrixB);
    free(matrixC);

    std::cout << "Total communication units per rank: " << 1.0*total_communication/strategy.P << std::endl;
    std::cout << "Total computation units: " << max_total_computation << std::endl;
    std::cout << "Max buffer size: " << max_buffer_size << std::endl;
    std::cout << "Local m = " << local_m << std::endl;
    std::cout << "Local n = " << local_n << std::endl;
    std::cout << "Local k = " << local_k << std::endl;
}

// dispatch to local call, BFS, or DFS as appropriate
void multiply(Interval& m, Interval& n, Interval& k, Interval& P, int step,
              const Strategy& strategy, double beta, int rank) {
    // current submatrices that are being computed
    Interval2D a_range(m, k);
    Interval2D b_range(k, n);
    Interval2D c_range(m, n);

    // For each of P processors remember which DFS bucket we are currently on
    std::vector<int> bucketA = matrixA->dfs_buckets(P);
    std::vector<int> bucketB = matrixB->dfs_buckets(P);
    std::vector<int> bucketC = matrixC->dfs_buckets(P);

    // Skip all buckets that are "before" the current submatrices. 
    // the relation submatrix1 <before> submatrix2 is defined in Interval2D.
    // Intuitively, this will skip all the buckets that are "above" or "on the left" 
    // of the current submatrices. We say "before" because whenever we split in DFS
    // sequentially, we always first start with the "above" submatrix 
    // (if the splitting is horizontal) or with the left one (if the splitting is vertical).
    // which explains the name of the relation "before".
    matrixA->update_buckets(P, a_range);
    matrixB->update_buckets(P, b_range);
    matrixC->update_buckets(P, c_range);

    if (strategy.final_step(step))
        local_multiply(m.length(), n.length(), k.length(), beta);
    else {
        if (strategy.bfs_step(step))
            BFS(m, n, k, P, step, strategy, beta, rank);
        else
            DFS(m, n, k, P, step, strategy, beta, rank);
    }

    // Revert the buckets pointers to their previous values.
    matrixA->set_dfs_buckets(P, bucketA);
    matrixB->set_dfs_buckets(P, bucketB);
    matrixC->set_dfs_buckets(P, bucketC);
}

void local_multiply(int m, int n, int k, double beta) {
    long long comp = m * n * k;
    if (comp > max_total_computation) {
        max_total_computation = comp;
        local_m = m;
        local_n = n;
        local_k = k;
    }
}

/*
  In each DFS step, one of the dimensions is split, and each of the subproblems is solved
  sequentially by all P processors.
*/
void DFS(Interval& m, Interval& n, Interval& k, Interval& P, int step,
         const Strategy& strategy, double beta, int rank) {
    // split the dimension but not the processors, all P processors are taking part
    // in each recursive call.
    if (strategy.split_m(step)) {
        for (int M = 0; M < strategy.divisor(step); ++M) {
            Interval newm = m.subinterval(strategy.divisor(step), M);
            multiply(newm, n, k, P, step+1, strategy, beta, rank);
        }
        return;
    }

    if (strategy.split_n(step)) {
        for (int N = 0; N < strategy.divisor(step); ++N) {
            Interval newn = n.subinterval(strategy.divisor(step), N);
            multiply(m, newn, k, P, step+1, strategy, beta, rank);
        }
        return;
    }

    // if divided by k, then the result of each subproblem is just a partial result for C
    // which should all be summed up. We solve this by letting beta parameter be 1
    // in the recursive calls that follow so that dgemm automatically adds up the subsequent
    // results to the previous partial results of C.
    if (strategy.split_k(step)) {
        for (int K = 0; K < strategy.divisor(step); ++K) {
            Interval newk = k.subinterval(strategy.divisor(step), K);
            multiply(m, n, newk, P, step+1, strategy, (K==0)&&(beta==0) ? 0 : 1, rank);
        }
        return;
    }
}

template<typename T>
T which_is_expanded(T A, T B, T C, const Strategy& strategy, size_t step) {
    // divn > 1 => divm==divk==1 => matrix A has not been splitted
    // therefore it is expanded (in the communication step of BFS)
    if (strategy.split_n(step))
        return A;

    // divm > 1 => divk==divn==1 => matrix B has not been splitted
    // therefore it is expanded (in the communication step of BFS)
    if (strategy.split_m(step))
        return B;

    // divk > 1 => divm==divn==1 => matrix C has not been splitted
    // therefore it is expanded (in the reduction step of BFS)
    if (strategy.split_k(step))
        return C;

    // this should never happen
    return C;
}

void BFS(Interval& m, Interval& n, Interval& k, Interval& P, int step,
         const Strategy& strategy, double beta, int rank) {
    int div = strategy.divisor(step);
    int divm = strategy.divisor_m(step);
    int divn = strategy.divisor_n(step);
    int divk = strategy.divisor_k(step);

    // processor subinterval which the current rank belongs to
    int partition_idx = P.partition_index(div, rank);
    Interval newP = P.subinterval(div, partition_idx);
    // intervals of M, N and K that the current rank is in charge of,
    // together with other ranks from its group.
    // (see the definition of group and offset below)
    Interval newm = m.subinterval(divm, divm>1 ? partition_idx : 0);
    Interval newn = n.subinterval(divn, divn>1 ? partition_idx : 0);
    Interval newk = k.subinterval(divk, divk>1 ? partition_idx : 0);

    int offset = rank - newP.first();
    int gp = (rank - P.first()) / newP.length();

    /*
     * size_before_expansion:
         maps rank i from interval P to the vector [bucket1.size(), bucket2.size()...]
         containing buckets which are inside "range" that rank i owns

     * total_before_expansion:
         maps rank i from interval P to the sum of all buckets inside size_before_expansion[i]

     * size_after_expansion:
         maps rank i from interval newP to the vector of [bucket1.size(), bucket2.size()...]
         but each bucket here is expanded, i.e. each bucket size in this vector
         is actually the sum of the sizes of this bucket in all the ranks 
         from the communication ring of the current rank.

     * total_after_expansion:
         maps rank i from interval P to the sum of all buckets inside size_after_expansion[i]
    */
    std::vector<std::vector<int>> size_before_expansion(P.length());
    std::vector<int> total_before_expansion(P.length());
    std::vector<std::vector<int>> size_after_expansion(newP.length());
    std::vector<int> total_after_expansion(newP.length());

    /*
     * this gives us the 2D interval of the matrix that will be expanded:
         if divm > 1 => matrix B expanded => Interval2D(k, n)
         if divn > 1 => matrix A expanded => Interval2D(m, k)
         if divk > 1 => matrix C expanded => Interval2D(m, n)
    */
    Interval row_copy = which_is_expanded(m, k, m, strategy, step);
    Interval col_copy = which_is_expanded(k, n, n, strategy, step);
    Interval2D range(row_copy, col_copy);

    /*
     * this gives us a matrix that is expanded:
         if divm > 1 => matrix B is expanded
         if divn > 1 => matrix A is expanded
         if divk > 1 => matrix C is expanded
    */
    CarmaMatrix* expanded_mat = which_is_expanded(matrixA, matrixB, matrixC, strategy, step);
    // gets the buffer sizes before and after expansion.
    // this still does not modify the buffer sizes inside layout
    // it just tells us what they would be.
    expanded_mat->buffers_before_expansion(P, range, 
            size_before_expansion, total_before_expansion);

    expanded_mat->buffers_after_expansion(P, newP, 
            size_before_expansion, total_before_expansion,
            size_after_expansion, total_after_expansion);

    // increase the buffer sizes before the recursive call
    expanded_mat->set_sizes(newP, size_after_expansion);
    // this is the sum of sizes of all the buckets after expansion
    // that the current rank will own.
    // which is also the size of the matrix after expansion
    long long new_size = total_after_expansion[offset];
    max_buffer_size = std::max(max_buffer_size, new_size);
    int received_volume = new_size - total_before_expansion[rank - P.first()];
    total_communication += received_volume;

    // invoke recursion with the new communicator containing ranks from newP
    // observe that we have only one recursive call here (we are not entering a loop
    // of recursive calls as in DFS steps since the current rank will only enter
    // into one recursive call since ranks are split).
    multiply(newm, newn, newk, newP, step+1, strategy, beta, rank);

    // after the memory is freed, the buffer sizes are back to the previous values 
    // (the values at the beginning of this BFS step)
    expanded_mat->set_sizes(newP, size_before_expansion, newP.first() - P.first());
}
