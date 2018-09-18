#include "multiply.hpp"

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
void multiply(CosmaMatrix& matrixA, CosmaMatrix& matrixB, CosmaMatrix& matrixC,
              const Strategy& strategy, MPI_Comm comm, double beta, bool one_sided_communication) {
    Interval mi = Interval(0, strategy.m-1);
    Interval ni = Interval(0, strategy.n-1);
    Interval ki = Interval(0, strategy.k-1);
    Interval Pi = Interval(0, strategy.P-1);

    PE(preprocessing_communicators);
    std::unique_ptr<communicator> cosma_comm;
    if (one_sided_communication) {
        cosma_comm = std::make_unique<one_sided_communicator>(&strategy, comm);

    } else {
        cosma_comm = std::make_unique<two_sided_communicator>(&strategy, comm);
    }
    PL();

    if (!cosma_comm->is_idle()) {
        multiply(matrixA, matrixB, matrixC,
                 mi, ni, ki, Pi, 0, strategy, *cosma_comm, beta);
    }

    if (cosma_comm->rank() == 0) {
        PP();
    }

}

// dispatch to local call, BFS, or DFS as appropriate
void multiply(CosmaMatrix& matrixA, CosmaMatrix& matrixB, CosmaMatrix& matrixC,
              Interval& m, Interval& n, Interval& k, Interval& P,
              size_t step, const Strategy& strategy,
              communicator& comm, double beta) {
    PE(multiply_layout);
    // current submatrices that are being computed
    Interval2D a_range(m, k);
    Interval2D b_range(k, n);
    Interval2D c_range(m, n);

    // For each of P processors remember which DFS bucket we are currently on
    std::vector<int> bucketA = matrixA.dfs_buckets(P);
    std::vector<int> bucketB = matrixB.dfs_buckets(P);
    std::vector<int> bucketC = matrixC.dfs_buckets(P);

    // Skip all buckets that are "before" the current submatrices.
    // the relation submatrix1 <before> submatrix2 is defined in Interval2D.
    // Intuitively, this will skip all the buckets that are "above" or "on the left"
    // of the current submatrices. We say "before" because whenever we split in DFS
    // sequentially, we always first start with the "above" submatrix
    // (if the splitting is horizontal) or with the left one (if the splitting is vertical).
    // which explains the name of the relation "before".
    matrixA.update_buckets(P, a_range);
    matrixB.update_buckets(P, b_range);
    matrixC.update_buckets(P, c_range);

    // This iterates over the skipped buckets and sums up their sizes,
    // and increases the pointer of the current matrix for the offset
    int offsetA = matrixA.shift(bucketA[comm.relative_rank(P)]);
    int offsetB = matrixB.shift(bucketB[comm.relative_rank(P)]);
    int offsetC = matrixC.shift(bucketC[comm.relative_rank(P)]);
    PL();

    if (strategy.final_step(step))
        local_multiply(matrixA, matrixB, matrixC, m.length(), n.length(), k.length(), beta);
    else {
        if (strategy.bfs_step(step))
            BFS(matrixA, matrixB, matrixC, m, n, k, P, step, strategy, comm, beta);
        else
            DFS(matrixA, matrixB, matrixC, m, n, k, P, step, strategy, comm, beta);
    }
    PE(multiply_layout);

    // shift the pointers of the current matrix back
    matrixA.unshift(offsetA);
    matrixB.unshift(offsetB);
    matrixC.unshift(offsetC);

    // Revert the buckets pointers to their previous values.
    matrixA.set_dfs_buckets(P, bucketA);
    matrixB.set_dfs_buckets(P, bucketB);
    matrixC.set_dfs_buckets(P, bucketC);
    PL();
}

void printMat(int m, int n, double*A, char label) {
    std::cout << "Matrix " << label << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[j * m + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void local_multiply(CosmaMatrix& matrixA, CosmaMatrix& matrixB, CosmaMatrix& matrixC, int m, int n, int k, double beta) {
    char N = 'N';
    double one = 1.;
#ifdef DEBUG
    double zero = 0.;
    if (beta > 0) {
        std::cout << "C (before) = " << std::endl;
        printMat(m, n, matrixC.current_matrix(), 'C');
        auto C_partial = std::unique_ptr<double[]>(new double[m * n]);
        dgemm_(&N, &N, &m, &n, &k, &one, matrixA.current_matrix(), &m, matrixB.current_matrix(), &k, &zero, C_partial.get(), &m);
        std::cout << "C (partial) = " << std::endl;
        printMat(m, n, C_partial.get(), 'C');
    }
#endif
    PE(multiply_computation);
#ifdef COSMA_HAVE_GPU
    gpu_dgemm_(matrixA.current_matrix(), matrixB.current_matrix(), matrixC.current_matrix(),
        matrixA.device_buffer_ptr(), matrixB.device_buffer_ptr(), matrixC.device_buffer_ptr(),
        m, n, k, 1.0, beta);
#else
    dgemm_(&N, &N, &m, &n, &k, &one, matrixA.current_matrix(), &m, matrixB.current_matrix(), &k, &beta, matrixC.current_matrix(), &m);
#endif
    // dgemm_(&N, &N, &m, &n, &k, &one, matrixA.current_matrix(), &m, matrixB.current_matrix(), &k, &beta, matrixC.current_matrix(), &m);
    PL();
#ifdef DEBUG
    std::cout << "After multiplication: " << std::endl;
    std::cout << "beta = " << beta << std::endl;
    printMat(m, k, matrixA.current_matrix(), 'A');
    printMat(k, n, matrixB.current_matrix(), 'B');
    printMat(m, n, matrixC.current_matrix(), 'C');
#endif
}

/*
 In each DFS step, one of the dimensions is split, and each of the subproblems is solved
 sequentially by all P processors.
 */
void DFS(CosmaMatrix& matrixA, CosmaMatrix& matrixB, CosmaMatrix& matrixC,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, communicator& comm, double beta) {
    // split the dimension but not the processors, all P processors are taking part
    // in each recursive call.
    if (strategy.split_m(step)) {
        for (int M = 0; M < strategy.divisor(step); ++M) {
            Interval newm = m.subinterval(strategy.divisor(step), M);
            multiply(matrixA, matrixB, matrixC, newm, n, k, P, step+1, strategy, 
                    comm, beta);
        }
        return;
    }

    if (strategy.split_n(step)) {
        for (int N = 0; N < strategy.divisor(step); ++N) {
            Interval newn = n.subinterval(strategy.divisor(step), N);
            multiply(matrixA, matrixB, matrixC, m, newn, k, P, step+1, strategy, 
                     comm, beta);
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
            multiply(matrixA, matrixB, matrixC, m, n, newk, P, step+1, strategy, 
                    comm, (K==0)&&(beta==0) ? 0 : 1);
        }
        return;
    }
}

template<typename T>
T which_is_expanded(T&& A, T&& B, T&& C, const Strategy& strategy, size_t step) {
    // divn > 1 => divm==divk==1 => matrix A has not been splitted
    // therefore it is expanded (in the communication step of BFS)
    if (strategy.split_n(step))
        return std::forward<T>(A);

    // divm > 1 => divk==divn==1 => matrix B has not been splitted
    // therefore it is expanded (in the communication step of BFS)
    if (strategy.split_m(step))
        return std::forward<T>(B);

    // divk > 1 => divm==divn==1 => matrix C has not been splitted
    // therefore it is expanded (in the reduction step of BFS)
    if (strategy.split_k(step))
        return std::forward<T>(C);

    // this should never happen
    return std::forward<T>(C);
}

/*
 In BFS step one of the dimensions is split into "div" pieces.
 Also, ranks P are split into "div" groups of "newP" processors
 such that each group of ranks is in charge of one piece of the split matrix.

 * if m split:
 Split matrix A and copy matrix B such that each of the "div" groups with newP processors
 contains the whole matrix B (that was previously owned by P processors).
 Communication is necessary since we want that newP<P processors own what was previously
 owned by P processors After the communication, each group of processors will contain
 identical data of matrix B in exactly the same order in all groups.

 * if n split:
 Split matrix B and copy matrix A such that each of the "div" groups with newP processors
 contains the whole matrix A (that was previously owned by P processors).
 Communication is necessary since we want that newP<P processors own what was previously
 owned by P processors After the communication, each group of processors will contain
 identical data of matrix A in exactly the same order in all groups.

 * if k split:
 Split both matrices A and B (since both of these matrices own dimension k).
 After the recursive call, each of "div" groups with newP processors will own
 a partial result of matrix C which should all be summed up (reduced) and splitted
 equally among all P processors. Thus, here we first sum up all the partial results
 that are owned by newP processors, and then we split it equally among P processors.
 While in the previous two cases we had to expand local matrices (since newP processors
 had to own what was previously owned by P processors) here we have the opposite - P ranks
 should own what was previously owned by newP ranks - thus local matrices are shrinked.
 */
void BFS(CosmaMatrix& matrixA, CosmaMatrix& matrixB, CosmaMatrix& matrixC,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, communicator& comm, double beta) {
    int divisor = strategy.divisor(step);
    int divisor_m = strategy.divisor_m(step);
    int divisor_n = strategy.divisor_n(step);
    int divisor_k = strategy.divisor_k(step);
    // processor subinterval which the current rank belongs to
    int partition_idx = P.partition_index(divisor, comm.rank());
    Interval newP = P.subinterval(divisor, partition_idx);
    // intervals of M, N and K that the current rank is in charge of,
    // together with other ranks from its group.
    // (see the definition of group and offset below)
    Interval newm = m.subinterval(divisor_m, divisor_m>1 ? partition_idx : 0);
    Interval newn = n.subinterval(divisor_n, divisor_n>1 ? partition_idx : 0);
    Interval newk = k.subinterval(divisor_k, divisor_k>1 ? partition_idx : 0);

    PE(multiply_layout);
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
    CosmaMatrix& expanded_mat = which_is_expanded(matrixA, matrixB, matrixC, strategy, step);
    // gets the buffer sizes before and after expansion.
    // this still does not modify the buffer sizes inside layout
    // it just tells us what they would be.
    expanded_mat.buffers_before_expansion(P, range,
                                          size_before_expansion, total_before_expansion);

    expanded_mat.buffers_after_expansion(P, newP,
                                         size_before_expansion, total_before_expansion,
                                         size_after_expansion, total_after_expansion);

    // increase the buffer sizes before the recursive call
    expanded_mat.set_sizes(newP, size_after_expansion);
    // this is the sum of sizes of all the buckets after expansion
    // that the current rank will own.
    // which is also the size of the matrix after expansion
    int new_size = total_after_expansion[comm.relative_rank(newP)];

    int buffer_idx = expanded_mat.buffer_index();
    expanded_mat.advance_buffer();

    double* original_matrix = expanded_mat.current_matrix();
    double* expanded_matrix = expanded_mat.buffer_ptr();
    double* reshuffle_buffer = expanded_mat.reshuffle_buffer_ptr();

    // pack the data for the next recursive call
    expanded_mat.set_current_matrix(expanded_matrix);
    PL();

    PE(multiply_communication_copy);
    // if divided along m or n then copy original matrix inside communication ring
    // to get the expanded matrix (all ranks inside communication ring should own
    // exactly the same data in the expanded matrix.
    if (strategy.split_m(step) || strategy.split_n(step)) {
        // copy the matrix that wasn't divided in this step
        comm.copy(P, original_matrix, expanded_matrix, reshuffle_buffer,
                  size_before_expansion, total_before_expansion, new_size, step);
    }
    PL();

    // if division by k, and we are in the branch where beta > 0, then
    // reset beta to 0, but keep in mind that on the way back from the recursion
    // we will have to sum the result with the local data in C
    // this is necessary since reduction happens AFTER the recursion
    // so we cannot pass beta = 1 if the data is not present there BEFORE the recursion.
    int new_beta = beta;
    if (strategy.split_k(step) && beta > 0) {
        new_beta = 0;
    }

    multiply(matrixA, matrixB, matrixC, newm, newn, newk, newP, step+1, strategy, comm, new_beta);
    // revert the current matrix
    expanded_mat.set_buffer_index(buffer_idx);
    expanded_mat.set_current_matrix(original_matrix);

    PE(multiply_communication_reduce);
    // if division by k do additional reduction of C
    if (strategy.split_k(step)) {
        double* reduce_buffer = expanded_mat.reduce_buffer_ptr();
        comm.reduce(P, expanded_matrix, original_matrix, 
                reshuffle_buffer, reduce_buffer,
                size_before_expansion, total_before_expansion, 
                size_after_expansion, total_after_expansion, beta, step);
    }
    PL();

    PE(multiply_layout);
    // after the memory is freed, the buffer sizes are back to the previous values
    // (the values at the beginning of this BFS step)
    expanded_mat.set_sizes(newP, size_before_expansion, newP.first() - P.first());
    PL();
}
