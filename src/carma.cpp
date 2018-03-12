#include "carma.h"

CarmaMatrix* matrixA;
CarmaMatrix* matrixB;
CarmaMatrix* matrixC;

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
void multiply(CarmaMatrix* A, CarmaMatrix* B, CarmaMatrix *C,
        int m, int n, int k, int P, int r,
        std::string::const_iterator patt,
        std::vector<int>::const_iterator divPatt) {
    PE("multiply");
    Interval mi = Interval(0, m-1);
    Interval ni = Interval(0, n-1);
    Interval ki = Interval(0, k-1);
    Interval Pi = Interval(0, P-1);

    matrixA = A;
    matrixB = B;
    matrixC = C;

    multiply(A->matrix_pointer(), B->matrix_pointer(), C->matrix_pointer(), 
            mi, ni, ki, Pi, r, patt, divPatt, 0.0, MPI_COMM_WORLD);

    PL("multiply");

#ifdef CARMA_HAVE_PROFILING
    for (size_t rank = 0; rank < P; ++rank) {
        if (rank == getRank()) {
            std::cout << "RANK " << rank << "\n";
            PP();
            std::cout << "\n\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}

// dispatch to local call, BFS, or DFS as appropriate
void multiply(double *A, double *B, double *C,
    Interval m, Interval n, Interval k, Interval P, int r,
    std::string::const_iterator patt,
    std::vector<int>::const_iterator divPatt, double beta,
    MPI_Comm comm) {

    PE("layout-overhead", "multiply");
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

    // This iterates over the skipped buckets and sums up their sizes so that
    // we know how many elements we should skip. 
    int offsetA = matrixA->offset(bucketA[getRank()-P.first()]);
    int offsetB = matrixB->offset(bucketB[getRank()-P.first()]);
    int offsetC = matrixC->offset(bucketC[getRank()-P.first()]);
    PL("layout-overhead");

    if (r == 0) {
        if(!P.only_one()) {
            printf("Error: reached r=0 with more than one processor\n");
            exit(-1);
        }
        local_multiply(A+offsetA, B+offsetB, C+offsetC, m.length(), n.length(), k.length(), beta);
    } else {
        if (patt[0] == 'B' || patt[0] == 'b') {
            BFS(A+offsetA, B+offsetB, C+offsetC, m, n, k, P, r, 
                    divPatt[0], divPatt[1], divPatt[2], patt+1, divPatt+3, beta, comm);
        } else if (patt[0] == 'D' || patt[0] == 'd') {

            DFS(A+offsetA, B+offsetB, C+offsetC, m, n, k, P, r, 
                    divPatt[0], divPatt[1], divPatt[2], patt+1, divPatt+3, beta, comm);

        } else {
            printf("Error: unrecognized type of step: %c\n", patt[0]);
        }
    }
    PE("layout-overhead", "multiply");
    // Revert the buckets pointers to their previous values.
    matrixA->set_dfs_buckets(P, bucketA);
    matrixB->set_dfs_buckets(P, bucketB);
    matrixC->set_dfs_buckets(P, bucketC);
    PL("layout-overhead");
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

void local_multiply(double *A, double *B, double *C, int m, int n, int k, double beta) {
    char N = 'N';
    double one = 1.;
#ifdef DEBUG
    double zero = 0.;
    if (beta > 0) {
        std::cout << "C (before) = " << std::endl;
        printMat(m, n, C, 'C');
        double* C_partial = (double*) malloc(sizeof(double) * m * n);
        dgemm_( &N, &N, &m, &n, &k, &one, A, &m, B, &k, &zero, C_partial, &m );
        std::cout << "C (partial) = " << std::endl;
        printMat(m, n, C_partial, 'C');
    }
#endif
    PE("computation", "multiply");
    dgemm_( &N, &N, &m, &n, &k, &one, A, &m, B, &k, &beta, C, &m );
    PL("computation");
#ifdef DEBUG
    std::cout << "After multiplication: " << std::endl;
    std::cout << "beta = " << beta << std::endl;
    printMat(m, k, A, 'A');
    printMat(k, n, B, 'B');
    printMat(m, n, C, 'C');
#endif
}

/*
  In each DFS step, one of the dimensions is split, and each of the subproblems is solved
  sequentially by all P processors.
*/
void DFS(double *A, double *B, double *C,
    Interval m, Interval n, Interval k, Interval P, int r, int divm, int divn, int divk,
    std::string::const_iterator patt,
    std::vector<int>::const_iterator divPatt, double beta, MPI_Comm comm) {

#ifdef DEBUG
    std::cout << "DFS with " << m << " " << n << " " << k << std::endl;
#endif

    // split the dimension but not the processors, all P processors are taking part
    // in each recursive call.
    if (divm > 1) {
        for (int M = 0; M < divm; ++M) {
            Interval newm = m.subinterval(divm, M);
            multiply(A, B, C, newm, n, k, P, r - 1, patt, divPatt, beta, comm);
        }
        return;
    }

    if (divn > 1) {
        for (int N = 0; N < divn; ++N) {
            Interval newn = n.subinterval(divn, N);
            multiply(A, B, C, m, newn, k, P, r - 1, patt, divPatt, beta, comm);
        }
        return;
    }

    // if divided by k, then the result of each subproblem is just a partial result for C
    // which should all be summed up. We solve this by letting beta parameter be 1
    // in the recursive calls that follow so that dgemm automatically adds up the subsequent
    // results to the previous partial results of C.
    if (divk > 1) {
        for (int K = 0; K < divk; ++K) {
            Interval newk = k.subinterval(divk, K);
            multiply(A, B, C, m, n, newk, P, r - 1, patt, divPatt, (K==0)&&(beta==0) ? 0 : 1, comm);
        }
        return;
    } 
}

template<typename T>
T which_is_expanded(T A, T B, T C, int divm, int divn, int divk) {
    // divn > 1 => divm==divk==1 => matrix A has not been splitted
    // therefore it is expanded (in the communication step of BFS)
    if (divn > 1)
        return A;

    // divm > 1 => divk==divn==1 => matrix B has not been splitted
    // therefore it is expanded (in the communication step of BFS)
    if (divm > 1)
        return B;

    // divk > 1 => divm==divn==1 => matrix C has not been splitted
    // therefore it is expanded (in the reduction step of BFS)
    if (divk > 1)
        return C;

    // this should never happen
    return C;
}

template<typename T>
T expand_if_needed(T original, T expanded, int divm, int divn) {
    // if the dimensions of this matrix were split, then this matrix is not expanded since we
    // since CARMA assumes that before splitting the data already resides on the right ranks
    if (divm + divn > 2)
        return original;

    return expanded;
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
void BFS(double *A, double *B, double *C,
             Interval m, Interval n, Interval k, Interval P, int r, int divm, int divn, int divk,
             std::string::const_iterator patt,
             std::vector<int>::const_iterator divPatt, double beta, MPI_Comm comm) {
#ifdef DEBUG
    std::cout << "BFS with " << m << " " << n << " " << k 
        << " divm = " << divm << " divn = " << divn << " divk = " << divk << std::endl;
#endif

    int div = divm * divn * divk;
    // check if only 1 dimension is divided in this step
    // Diophantine equation (x y z + 2 = x + y + z, s.t. x, y, z >= 1) has only solutions
    // of the form (k, 1, 1), (1, k, 1) and (1, 1, k) where k >= 1
    if (div + 2 != divm + divn + divk) {
        std::cout << "In each step, only one dimension can be split. Aborting the application...\n";
        exit(-1);
    }

    // processor subinterval which the current rank belongs to
    int partition_idx = P.partition_index(div, getRank());
    Interval newP = P.subinterval(div, partition_idx);
    // intervals of M, N and K that the current rank is in charge of,
    // together with other ranks from its group.
    // (see the definition of group and offset below)
    Interval newm = m.subinterval(divm, divm>1 ? partition_idx : 0);
    Interval newn = n.subinterval(divn, divn>1 ? partition_idx : 0);
    Interval newk = k.subinterval(divk, divk>1 ? partition_idx : 0);

    /*
      We split P processors into div groups of newP.length() processors.
        * gp from [0..(div-1)] is the id of the group of the current rank
        * offset from [0..(newP.length()-1)] is the offset of current rank inside its group

      We now define the communication ring of the current processor as:
        i * newP.length() + offset, where i = 0..(div-1) and offset = getRank() - newP.first()
    */
    int offset = getRank() - newP.first();
    int gp = (getRank() - P.first()) / newP.length();

    // New communicator splitting P processors into div groups of newP.length() rocessors
    // This communicator will be used in the recursive call.
    MPI_Comm newcomm;
    MPI_Comm_split(comm, gp, offset, &newcomm);

    PE("layout-overhead", "multiply");
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
    Interval row_copy = which_is_expanded(m, k, m, divm, divn, divk);
    Interval col_copy = which_is_expanded(k, n, n, divm, divn, divk);
    Interval2D range(row_copy, col_copy);

    /*
     * this gives us a matrix that is expanded:
         if divm > 1 => matrix B is expanded
         if divn > 1 => matrix A is expanded
         if divk > 1 => matrix C is expanded
    */
    CarmaMatrix* expanded_mat = which_is_expanded(matrixA, matrixB, matrixC, divm, divn, divk);
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
    int new_size = total_after_expansion[offset];

    // new allocated space for the expanded matrix
    double* expanded_space = (double*) malloc(new_size * sizeof(double));

    // LM = M if M was not expanded
    // LM = expanded_space if M was expanded
    double* LA = expand_if_needed(A, expanded_space, divm, divk);
    double* LB = expand_if_needed(B, expanded_space, divk, divn);
    double* LC = expand_if_needed(C, expanded_space, divm, divn);

    // if divm > 1 => original_matrix=B, expanded_matrix=LB
    // if divn > 1 => original_matrix=A, expanded_matrix=LA
    // if divk > 1 => original_matrix=C, expanded_matrix=LC
    double* original_matrix = which_is_expanded(A, B, C, divm, divn, divk);
    double* expanded_matrix = which_is_expanded(LA, LB, LC, divm, divn, divk);
    PL("layout-overhead");

    PE("communication", "multiply");
    // if divided along m or n then copy original matrix inside communication ring
    // to get the expanded matrix (all ranks inside communication ring should own
    // exactly the same data in the expanded matrix.
    if (divm + divn > 2) {
        PE("copying", "communication");
        // copy the matrix that wasn't divided in this step
        copy_mat(div, P, newP, range, original_matrix, expanded_matrix,
                size_before_expansion, total_before_expansion, new_size, comm);
        PL("copying");
        /*
          observe that here we use the communicator "comm" and not "newcomm"
          this is because newcomm contains only ranks inside newP
          however, we need to communicate with all the ranks from the communication
          ring to make sure that the group of processors newP now owns everything 
          that was previously owned by P processors and can thus continue
          the execution independently of the other processors in "comm"
        */
    }
    PL("communication");
    // invoke recursion with the new communicator containing ranks from newP
    // observe that we have only one recursive call here (we are not entering a loop
    // of recursive calls as in DFS steps since the current rank will only enter
    // into one recursive call since ranks are split).
    multiply(LA, LB, LC, newm, newn, newk, newP, r-1, patt, divPatt, beta, newcomm);

    PE("communication", "multiply");
    // if division by k do additional reduction of C
    if (divk > 1) {
        PE("reduction", "communication");
        reduce(div, P, newP, range, expanded_matrix, original_matrix, size_before_expansion, 
               total_before_expansion, size_after_expansion, total_after_expansion, comm);
        PL("reduction");
    }
    PL("communication");

    PE("layout-overhead", "multiply");
    // after the memory is freed, the buffer sizes are back to the previous values 
    // (the values at the beginning of this BFS step)
    expanded_mat->set_sizes(newP, size_before_expansion, newP.first() - P.first());
    PL("layout-overhead");

    free(expanded_matrix);
    MPI_Comm_free(&newcomm);
}
