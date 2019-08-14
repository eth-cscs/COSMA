#include <cosma/local_multiply.hpp>
#include <cosma/mpi_mapper.hpp>
#include <cosma/multiply.hpp>

#include <complex>
#include <random>

using namespace cosma;

template <typename T>
void fill_matrix(T* ptr, size_t size) {
    std::random_device dev;                        // seed
    std::mt19937 rng(dev());                       // generator
    std::uniform_real_distribution<T> dist(1.); // distribution

    for (unsigned i = 0; i < size; ++i) {
        ptr[i] = T{dist(rng)};
    }
}

template <typename T>
void fill_matrix(std::complex<T>* ptr, size_t size) {
    std::random_device dev;                        // seed
    std::mt19937 rng(dev());                       // generator
    std::uniform_real_distribution<T> dist(1.); // distribution

    for (unsigned i = 0; i < size; ++i) {
        ptr[i] = std::complex<T>{dist(rng), dist(rng)};
    }
}

template <typename Scalar>
bool run(Strategy &s,
         context<Scalar> &ctx,
         MPI_Comm comm = MPI_COMM_WORLD,
         bool overlap = false) {
    constexpr auto epsilon = std::numeric_limits<float>::epsilon();

    int rank;
    int size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    auto mpi_type = cosma::mpi_mapper<Scalar>::getType();

    s.overlap_comm_and_comp = overlap;
    int m = s.m;
    int n = s.n;
    int k = s.k;
    int P = s.P;

    // Declare A,B and C COSMA matrices objects
    // CosmaMatrix<Scalar> A(ctx, 'A', s, rank);
    // CosmaMatrix<Scalar> B(ctx, 'B', s, rank);
    // CosmaMatrix<Scalar> C(ctx, 'C', s, rank);

    CosmaMatrix<Scalar> A('A', s, rank);
    CosmaMatrix<Scalar> B('B', s, rank);
    CosmaMatrix<Scalar> C('C', s, rank);

    // initial sizes
    auto sizeA = A.matrix_size();
    auto sizeB = B.matrix_size();
    auto sizeC = C.matrix_size();

    // fill the matrices with random data
    srand48(rank);
    fill_matrix(A.matrix_pointer(), sizeA);
    fill_matrix(B.matrix_pointer(), sizeB);

#ifdef DEBUG
    std::cout << "Initial data in A and B:" << std::endl;
    for (int i = 0; i < P; i++) {
        if (rank == i) {
            printf("(%d) A: ", i);
            for (auto j = 0; j < sizeA; j++)
                printf("%5.3f ", A.matrix_pointer()[j]);
            printf("\n");

            printf("(%d) B: ", i);
            for (auto j = 0; j < sizeB; j++)
                printf("%5.3f ", B.matrix_pointer()[j]);
            printf("\n");
        }
        MPI_Barrier(comm);
    }
#endif // DEBUG

    bool isOK;

    // Then rank0 ask for other ranks data
    std::vector<Scalar> As, Bs;
    if (rank == 0) {
        As = std::vector<Scalar>(m * k);
        std::memcpy(As.data(), A.matrix_pointer(), A.matrix_size()*sizeof(Scalar));
        Bs = std::vector<Scalar>(k * n);
        std::memcpy(Bs.data(), B.matrix_pointer(), B.matrix_size()*sizeof(Scalar));

        int offsetA = sizeA;
        int offsetB = sizeB;

        for (int i = 1; i < P; i++) {
            int receive_size_A = A.matrix_size(i);
            int receive_size_B = B.matrix_size(i);
            // Rank 0 receive data
            MPI_Recv(As.data() + offsetA,
                     receive_size_A,
                     mpi_type,
                     i,
                     0,
                     comm,
                     MPI_STATUSES_IGNORE);
            MPI_Recv(Bs.data() + offsetB,
                     receive_size_B,
                     mpi_type,
                     i,
                     0,
                     comm,
                     MPI_STATUSES_IGNORE);

            offsetA += receive_size_A;
            offsetB += receive_size_B;
        }
    }
    // Rank i send data
    if (rank > 0) {
        MPI_Send(A.matrix_pointer(), sizeA, mpi_type, 0, 0, comm);
        MPI_Send(B.matrix_pointer(), sizeB, mpi_type, 0, 0, comm);
    }

    MPI_Barrier(comm);

    // Then rank 0 must reorder data locally
    std::vector<Scalar> globA;
    std::vector<Scalar> globB;
    std::vector<Scalar> globCcheck;
    if (rank == 0) {
        globA.resize(m * k);
        globB.resize(k * n);
        globCcheck.resize(m * n);
        int offsetA = 0;
        int offsetB = 0;

        for (int i = 0; i < P; i++) {
            int local_size_A = A.matrix_size(i);
            int local_size_B = B.matrix_size(i);

            for (int j = 0; j < local_size_A; j++) {
                int y, x;
                std::tie(y, x) = A.global_coordinates(j, i);
                if (y >= 0 && x >= 0) {
                    globA.at(x * m + y) = As.at(offsetA + j);
                }
            }
            for (int j = 0; j < local_size_B; j++) {
                int y, x;
                std::tie(y, x) = B.global_coordinates(j, i);
                // std::cout << "Mapped successfully!\n";
                if (y >= 0 && x >= 0) {
                    // globB.at(x*n+y)=Bs.at(i*sizeB+j);
                    // std::cout << "Retrieved Bs value successfully!\n";
                    globB.at(x * k + y) = Bs.at(offsetB + j);
                }
            }

            offsetA += local_size_A;
            offsetB += local_size_B;
        }
        // Now compute the result
        cosma::local_multiply(globA.data(),
                              globB.data(),
                              globCcheck.data(),
                              m,
                              n,
                              k,
                              Scalar{1.0},
                              Scalar{0.0});
#ifdef DEBUG
        std::cout << "Complete matrix A: " << std::endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                std::cout << globA[j * m + i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        std::cout << "Complete matrix B: " << std::endl;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << globB[j * k + i] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "Complete matrix C: " << std::endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << globCcheck[j * m + i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
#endif
    }

    multiply(A, B, C, s, comm, Scalar{1}, Scalar{0});

    // Then rank0 ask for other ranks data
    std::vector<Scalar> Cs;
    if (rank == 0) {
        Cs = std::vector<Scalar>(m * n);
        std::memcpy(Cs.data(), C.matrix_pointer(), C.matrix_size()*sizeof(Scalar));

        int offsetC = sizeC;

        for (int i = 1; i < P; i++) {
            int receive_size_C = C.matrix_size(i);
            // Rank 0 receive data
            MPI_Recv(Cs.data() + offsetC,
                     receive_size_C,
                     mpi_type,
                     i,
                     0,
                     comm,
                     MPI_STATUSES_IGNORE);
            offsetC += receive_size_C;
        }
    }
    // Rank i send data
    if (rank > 0) {
        MPI_Send(C.matrix_pointer(), sizeC, mpi_type, 0, 0, comm);
    }

    MPI_Barrier(comm);

    // Then rank 0 must reorder data locally
    std::vector<Scalar> globC;
    if (rank == 0) {
        globC.resize(m * n);
        int offsetC = 0;

        for (int i = 0; i < P; i++) {
            int local_size_C = C.matrix_size(i);

            for (int j = 0; j < local_size_C; j++) {
                int y, x;
                std::tie(y, x) = C.global_coordinates(j, i);
                if (y >= 0 && x >= 0) {
                    globC.at(x * m + y) = Cs.at(offsetC + j);
                }
            }
            offsetC += local_size_C;
        }

        // Now Check result
        isOK = globCcheck.size() == globC.size();
        for (int i = 0; i < globC.size(); ++i) {
            isOK = isOK && (std::abs(globC[i] - globCcheck[i]) < epsilon);
        }

        if (!isOK) {
            std::cout << "Result is NOT OK" << std::endl;
            for (int i = 0; i < m * n; i++) {
                if (globCcheck[i] != globC[i]) {
                    int x = i % m;
                    int y = i / m;
                    int locidx, rank;
                    std::tie(locidx, rank) = C.local_coordinates(x, y);
                    std::cout << "global(" << x << ", " << y
                              << ") = (loc = " << locidx << ", rank = " << rank
                              << ") = " << globC.at(i) << " and should be "
                              << globCcheck.at(i) << std::endl;
                }
            }
        }
        else {
            std::cout <<"Result is OK"<<std::endl;
        }
    }
#ifdef DEBUG
    for (int i = 0; i < P; i++) {
        if (rank == i) {
            printf("(%d) A: ", i);
            for (auto j = 0; j < sizeA; j++)
                printf("%5.3f ", A.matrix_pointer()[j]);
            printf("\n");

            printf("(%d) B: ", i);
            for (auto j = 0; j < sizeB; j++)
                printf("%5.3f ", B.matrix_pointer()[j]);
            printf("\n");

            printf("(%d) C: ", i);
            for (auto j = 0; j < sizeC; j++)
                printf("%5.3f ", C.matrix_pointer()[j]);
            printf("\n");
        }
        MPI_Barrier(comm);
    }
#endif // DEBUG
    return rank > 0 || isOK;
}
