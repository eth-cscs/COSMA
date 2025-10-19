#include <cosma/one_sided_communicator.hpp>

#include <cosma/bfloat16.hpp>
#include <cosma/local_multiply.hpp>
#include <cosma/math_utils.hpp>
#include <cosma/mpi_mapper.hpp>
#include <cosma/profiler.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <stdlib.h>
#include <thread>
#include <tuple>
#include <vector>

namespace cosma {

namespace one_sided_communicator {

template <typename Scalar>
MPI_Win
create_window(MPI_Comm comm, Scalar *pointer, size_t size, bool no_locks) {
    MPI_Info info;
    MPI_Info_create(&info);
    if (no_locks) {
        MPI_Info_set(info, "no_locks", "true");
    } else {
        MPI_Info_set(info, "no_locks", "false");
    }
    MPI_Info_set(info, "accumulate_ops", "same_op");
    MPI_Info_set(info, "accumulate_ordering", "none");

    MPI_Win win;
    MPI_Win_create(
        pointer, size * sizeof(Scalar), sizeof(Scalar), info, comm, &win);

    MPI_Info_free(&info);

    return win;
}

template <typename Scalar>
void copy(MPI_Comm comm,
          int rank,
          int div,
          Interval &P,
          Scalar *in,
          Scalar *out,
          Scalar *reshuffle_buffer,
          std::vector<std::vector<int>> &size_before,
          std::vector<int> &total_before,
          int total_after) {
    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);

    int relative_rank = rank - P.first();
    int local_size = total_before[relative_rank];

    MPI_Win win = create_window(comm, in, local_size, true);
    MPI_Win_fence(MPI_MODE_NOPRECEDE + MPI_MODE_NOPUT, win);

    int n_blocks = size_before[relative_rank].size();
    std::vector<int> rank_offset(div);

    auto mpi_type = mpi_mapper<Scalar>::getType();
    int displacement = 0;
    for (int block = 0; block < n_blocks; block++) {
        for (int rank = 0; rank < div; ++rank) {
            int target = P.locate_in_interval(div, rank, off);
            int b_size = size_before[target][block];

            MPI_Get(out + displacement,
                    b_size,
                    mpi_type,
                    rank,
                    rank_offset[rank],
                    b_size,
                    mpi_type,
                    win);

            rank_offset[rank] += b_size;
            displacement += b_size;
        }
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
    MPI_Win_free(&win);

#ifdef DEBUG
    std::cout << "Content of the copied matrix in rank " << rank
              << " is now: " << std::endl;
    for (int j = 0; j < rank_offset[gp]; j++) {
        std::cout << out[j] << ", ";
    }
    std::cout << std::endl;
#endif
}

template <typename Scalar>
void reduce(MPI_Comm comm,
            int rank,
            int div,
            Interval &P,
            Scalar *in,
            Scalar *out,
            Scalar *reshuffle_buffer,
            Scalar *reduce_buffer,
            std::vector<std::vector<int>> &c_current,
            std::vector<int> &c_total_current,
            std::vector<std::vector<int>> &c_expanded,
            std::vector<int> &c_total_expanded,
            Scalar beta) {
    auto mpi_type = mpi_mapper<Scalar>::getType();
    // int div = strategy_->divisor(step);
    // int gp, off;
    // std::tie(gp, off) = group_and_offset(P, div);
    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);

    int n_blocks = c_expanded[off].size();

    int target = P.locate_in_interval(div, gp, off);
    int local_size = c_total_current[target];

    // initilize C to 0 if beta = 0 since accumulate will do additions over
    // this array
    if (beta == Scalar{0}) {
        memset(out, 0, local_size * sizeof(Scalar));
    }

    MPI_Win win = create_window(comm, out, local_size, true);
    MPI_Win_fence(MPI_MODE_NOPRECEDE + MPI_MODE_NOSTORE, win);

    int displacement = 0;
    std::vector<int> rank_offset(div);
    // go through the communication ring
    for (int block = 0; block < n_blocks; ++block) {
        for (int i = 0; i < div; ++i) {
            int target = P.locate_in_interval(div, i, off);
            int b_size = c_current[target][block];

            MPI_Accumulate(in + displacement,
                           b_size,
                           mpi_type,
                           i,
                           rank_offset[i],
                           b_size,
                           mpi_type,
                           MPI_SUM,
                           win);

            displacement += b_size;
            rank_offset[i] += b_size;
        }
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
    MPI_Win_free(&win);
}

template <typename Scalar>
void comm_task_mn_split_polling(int divisor,
                                int gp,
                                Scalar *original_matrix,
                                Scalar *expanded_matrix,
                                Interval m,
                                Interval k,
                                std::vector<int> &displacements,
                                std::atomic_int &ready,
                                MPI_Comm comm) {
    PE(multiply_communication_other);
    // copy the matrix that wasn't divided in this step
    int local_size = m.length() * k.subinterval(divisor, gp).length();

    auto mpi_type = mpi_mapper<Scalar>::getType();
    MPI_Win win = create_window(comm, original_matrix, local_size, false);
    // MPI_Comm mpi_comm = comm.active_comm(step);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

    int dist = 1;
    while (dist < divisor) {
        int rank = (gp + dist) % divisor;
        int b_size = m.length() * k.subinterval(divisor, rank).length();

        PL();
        PE(multiply_communication_copy);
        MPI_Request req;
        MPI_Rget(expanded_matrix + m.length() * displacements[rank],
                 b_size,
                 mpi_type,
                 rank,
                 0,
                 b_size,
                 mpi_type,
                 win,
                 &req);

        int finished = false;
        while (!finished) {
            MPI_Test(&req, &finished, MPI_STATUS_IGNORE);
            if (!finished) {
                std::this_thread::yield();
            } else {
                ready++;
            }
        }
        PL();

        PE(multiply_communication_other);
        dist++;
    }

    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
    PL();
}

template <typename Scalar>
void comm_task_mn_split_busy_waiting(int divisor,
                                     int gp,
                                     Scalar *original_matrix,
                                     Scalar *expanded_matrix,
                                     Interval m,
                                     Interval k,
                                     std::vector<int> &displacements,
                                     std::atomic_int &ready,
                                     MPI_Comm comm) {
    // copy the matrix that wasn't divided in this step
    PE(multiply_communication_other);
    int local_size = m.length() * k.subinterval(divisor, gp).length();

    MPI_Win win = create_window(comm, original_matrix, local_size, false);

#ifdef DEBUG
    std::cout << "window content: " << std::endl;
    for (int i = 0; i < local_size; ++i) {
        std::cout << *(original_matrix + i) << ", ";
    }
    std::cout << std::endl;
#endif

    // MPI_Comm mpi_comm = comm.active_comm(step);
    // MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

    auto mpi_type = mpi_mapper<Scalar>::getType();
    int dist = 1;
    while (dist < divisor) {
        int rank = (gp + dist) % divisor;
        int b_size = m.length() * k.subinterval(divisor, rank).length();
        PL();
#ifdef DEBUG
        std::cout << "Getting a piece from rank " << rank << std::endl;
#endif
        PE(multiply_communication_copy);
        MPI_Get(expanded_matrix + m.length() * displacements[rank],
                b_size,
                mpi_type,
                rank,
                0,
                b_size,
                mpi_type,
                win);

        // flush completes the operation locally
        // but since this is a Get operation,
        // then it also means that after flush
        // it will also be completed remotely
        MPI_Win_flush_local(rank, win);
        PL();

        PE(multiply_communication_other);
        dist++;
        ready++;
    }

    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
    PL();
}

/* OVERLAP OF M SPLIT WITH OPENMP
 * template<typename Scalar>
    void overlap_m_split(context& ctx, MPI_Comm comm, int rank, int
divisor, CosmaMatrix& matrixA, CosmaMatrix& matrixB, CosmaMatrix& matrixC,
            Interval& m, Interval& n, Interval& k, Interval& P, Scalar beta)
{ PE(multiply_communication_copy);

        int gp, off;
        std::tie(gp, off) = P.locate_in_subinterval(divisor, rank);

        CosmaMatrix& expanded_mat = matrixB;
        int buffer_idx = expanded_mat.buffer_index();
        expanded_mat.advance_buffer();

        Scalar* original_matrix = expanded_mat.current_matrix();
        Scalar* expanded_matrix = expanded_mat.buffer_ptr();

        // interval of m that this rank owns from this step on
        Interval newm = m.subinterval(divisor, gp);

        // copy the matrix that wasn't divided in this step
        int local_size = k.length() * n.subinterval(divisor, gp).length();

        // offsets in the expanded matrix for each rank
        std::vector<int> displacements_n(divisor);
        int disp = 0;

        for (int rank = 0; rank < divisor; ++rank) {
            displacements_n[rank] = disp;
            disp += n.subinterval(divisor, rank).length();
        }
        // b: k * disp
        // c: newm * disp

        Scalar* prev_a = matrixA.current_matrix();
        Scalar* prev_b = expanded_matrix;
        Scalar* prev_c = matrixC.current_matrix();

        MPI_Win win = create_window(comm, original_matrix, local_size,
false);

        auto mpi_type = mpi_mapper<Scalar>::getType();

#ifdef DEBUG
        std::cout << "window content: " << std::endl;
        for (int i = 0; i < local_size; ++i) {
            std::cout << *(original_matrix + i) << ", ";
        }
        std::cout << std::endl;
#endif

        // MPI_Comm mpi_comm = comm.active_comm(step);
        // MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

#pragma omp parallel num_threads(2)
        {
#pragma omp single nowait
#pragma omp critical
        {
            // compute the piece that is already owned
            Scalar* pointer_b = original_matrix;
            Scalar* pointer_c = prev_c + newm.length() *
displacements_n[gp];

            matrixB.set_current_matrix(pointer_b);
            matrixC.set_current_matrix(pointer_c);

            PL();
            local_multiply(ctx, matrixA.current_matrix(),
matrixB.current_matrix(), matrixC.current_matrix(), newm.length(),
                    n.subinterval(divisor, gp).length(), k.length(), beta);
            PE(multiply_communication_copy);
        }

#pragma omp single nowait
        for (int dist = 1; dist < divisor; dist++) {
            int rank = (gp+dist)%divisor;
            int b_size = k.length() * n.subinterval(divisor, rank).length();

            MPI_Get(expanded_matrix + k.length() * displacements_n[rank],
b_size, mpi_type, rank, 0, b_size, mpi_type, win);

            // flush completes the operation locally
            // but since this is a Get operation,
            // then it also means that after flush
            // it will also be completed remotely
            MPI_Win_flush_local(rank, win);

#pragma omp task firstprivate(dist, divisor)
#pragma omp critical
                {
                    // Compute the piece that has arrived
                    Scalar* pointer_b = expanded_matrix + k.length() *
displacements_n[rank]; Scalar* pointer_c = prev_c + newm.length() *
displacements_n[rank];

                    matrixB.set_current_matrix(pointer_b);
                    matrixC.set_current_matrix(pointer_c);

                    PL();
                    local_multiply_cpu(matrixA.current_matrix(),
matrixB.current_matrix(), matrixC.current_matrix(), newm.length(),
                            n.subinterval(divisor, rank).length(),
k.length(), beta);
                    // local_multiply(ctx, matrixA.current_matrix(),
matrixB.current_matrix(),
                    //         matrixC.current_matrix(), newm.length(),
                    //         n.subinterval(divisor, rank).length(),
k.length(), beta); PE(multiply_communication_copy);
                }
            }
#pragma omp taskwait
        }

        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);

        expanded_mat.set_current_matrix(original_matrix);
        expanded_mat.set_buffer_index(buffer_idx);
        matrixC.set_current_matrix(prev_c);

        PL();
    }
*/
// ***********************************
//           DIVISION BY M
// ***********************************
template <typename Scalar>
void overlap_m_split(bool use_busy_waiting,
                     cosma_context<Scalar> *ctx,
                     MPI_Comm comm,
                     int rank,
                     int divisor,
                     CosmaMatrix<Scalar> &matrixA,
                     CosmaMatrix<Scalar> &matrixB,
                     CosmaMatrix<Scalar> &matrixC,
                     Interval &m,
                     Interval &n,
                     Interval &k,
                     Interval &P,
                     Scalar alpha,
                     Scalar beta) {
    PE(multiply_communication_other);
    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(divisor, rank);

    CosmaMatrix<Scalar> &expanded_mat = matrixB;
    int buffer_idx = expanded_mat.buffer_index();
    expanded_mat.advance_buffer();

    Scalar *original_matrix = expanded_mat.current_matrix();
    Scalar *expanded_matrix = expanded_mat.buffer_ptr();

    // interval of m that this rank owns from this step on
    Interval newm = m.subinterval(divisor, gp);

    // copy the matrix that wasn't divided in this step
    int local_size = k.length() * n.subinterval(divisor, gp).length();

    // offsets in the expanded matrix for each rank
    std::vector<int> displacements_n(divisor);
    int disp = 0;

    for (int rank = 0; rank < divisor; ++rank) {
        displacements_n[rank] = disp;
        disp += n.subinterval(divisor, rank).length();
    }
    // b: k * disp
    // c: newm * disp

    std::atomic_int ready(0);
    std::thread comm_thread(use_busy_waiting
                                ? comm_task_mn_split_busy_waiting<Scalar>
                                : comm_task_mn_split_polling<Scalar>,
                            divisor,
                            gp,
                            original_matrix,
                            expanded_matrix,
                            k,
                            n,
                            std::ref(displacements_n),
                            std::ref(ready),
                            comm);

    Scalar *prev_a = matrixA.current_matrix();
    Scalar *prev_b = expanded_matrix;
    Scalar *prev_c = matrixC.current_matrix();

    // compute the piece that is already owned
    Scalar *pointer_b = original_matrix;
    Scalar *pointer_c = prev_c + newm.length() * displacements_n[gp];

    matrixB.set_current_matrix(pointer_b);
    matrixC.set_current_matrix(pointer_c);

    PL();
    bool copy_c_back = true;
    local_multiply(ctx,
                   matrixA.current_matrix(),
                   matrixB.current_matrix(),
                   matrixC.current_matrix(),
                   newm.length(),
                   n.subinterval(divisor, gp).length(),
                   k.length(),
                   alpha,
                   beta,
                   copy_c_back);
    PE(multiply_communication_other);

    int dist = 1;
    while (dist < divisor) {
        while (ready > 0) {
            int idx = (gp + dist) % divisor;

            // Compute the piece that has arrived
            Scalar *pointer_b =
                expanded_matrix + k.length() * displacements_n[idx];
            Scalar *pointer_c = prev_c + newm.length() * displacements_n[idx];

            matrixB.set_current_matrix(pointer_b);
            matrixC.set_current_matrix(pointer_c);

            PL();
            local_multiply(ctx,
                           matrixA.current_matrix(),
                           matrixB.current_matrix(),
                           matrixC.current_matrix(),
                           newm.length(),
                           n.subinterval(divisor, idx).length(),
                           k.length(),
                           alpha,
                           beta,
                           copy_c_back);
            PE(multiply_communication_copy);
            ready--;
            dist++;
        }
    }

    expanded_mat.set_current_matrix(original_matrix);
    expanded_mat.set_buffer_index(buffer_idx);
    matrixC.set_current_matrix(prev_c);

    comm_thread.join();
    PL();
}

// ***********************************
//           DIVISION BY N
// ***********************************
template <typename Scalar>
void overlap_n_split(bool use_busy_waiting,
                     cosma_context<Scalar> *ctx,
                     MPI_Comm comm,
                     int rank,
                     int divisor,
                     CosmaMatrix<Scalar> &matrixA,
                     CosmaMatrix<Scalar> &matrixB,
                     CosmaMatrix<Scalar> &matrixC,
                     Interval &m,
                     Interval &n,
                     Interval &k,
                     Interval &P,
                     Scalar alpha,
                     Scalar beta) {
    PE(multiply_communication_other);
    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(divisor, rank);

    CosmaMatrix<Scalar> &expanded_mat = matrixA;

    int buffer_idx = expanded_mat.buffer_index();
    expanded_mat.advance_buffer();

    Scalar *original_matrix = expanded_mat.current_matrix();
    Scalar *expanded_matrix = expanded_mat.buffer_ptr();
    // expanded_mat.set_current_matrix(expanded_matrix);

    Scalar *prev_a = expanded_matrix;
    Scalar *prev_b = matrixB.current_matrix();

    Interval newn = n.subinterval(divisor, gp);

    int local_size = m.length() * k.subinterval(divisor, gp).length();

    // offsets in the expanded matrix for each rank
    std::vector<int> displacements_k(divisor);
    int disp_k = 0;
    for (int rank = 0; rank < divisor; ++rank) {
        displacements_k[rank] = disp_k;
        disp_k += k.subinterval(divisor, rank).length();
    }
    // a: m * disp

    // memory enough for the largest block
    // used to overlap communication and computation
    std::vector<Scalar> block_buffer(
        newn.length() * math_utils::int_div_up(k.length(), divisor));
    // std::cout << "block buffer size = " << block_buffer.size() <<
    // std::endl;

    std::atomic_int ready(1);
    std::thread comm_thread(use_busy_waiting
                                ? comm_task_mn_split_busy_waiting<Scalar>
                                : comm_task_mn_split_polling<Scalar>,
                            divisor,
                            gp,
                            original_matrix,
                            expanded_matrix,
                            m,
                            k,
                            std::ref(displacements_k),
                            std::ref(ready),
                            comm);

    int dist = 0;
    while (dist < divisor) {
        while (ready > 0) {
            int idx = (gp + dist) % divisor;

            // Compute the piece that has arrived
            Scalar *pointer_a =
                dist == 0
                    ? original_matrix
                    : (expanded_matrix + m.length() * displacements_k[idx]);
            // Scalar* pointer_b = switch_buffers ? buffer2.data() :
            // buffer1.data();
            Scalar *pointer_b = block_buffer.data();

            for (int col = 0; col < newn.length(); ++col) {
                int column_size = k.subinterval(divisor, idx).length();
                int start = displacements_k[idx] + k.length() * col;
                std::memcpy(pointer_b + col * column_size,
                            prev_b + start,
                            column_size * sizeof(Scalar));
            }

            matrixA.set_current_matrix(pointer_a);
            matrixB.set_current_matrix(pointer_b);

            Scalar new_beta = dist == 0 ? beta : Scalar(1);
            PL();
            bool copy_c_back = true;
            local_multiply(ctx,
                           matrixA.current_matrix(),
                           matrixB.current_matrix(),
                           matrixC.current_matrix(),
                           m.length(),
                           newn.length(),
                           k.subinterval(divisor, idx).length(),
                           alpha,
                           new_beta,
                           copy_c_back);
            PE(multiply_communication_other);

            dist++;
            ready--;
        }
    }
    comm_thread.join();

    // revert the current matrix
    expanded_mat.set_buffer_index(buffer_idx);
    expanded_mat.set_current_matrix(original_matrix);
    matrixB.set_current_matrix(prev_b);

    PL();
}

template <typename Scalar>
void comm_task_k_split(int divisor,
                       int gp,
                       int off,
                       int jump_size,
                       Scalar *expanded_matrix,
                       Scalar *recv_buffer,
                       Interval m,
                       Interval n,
                       Interval P,
                       std::vector<int> &displacements,
                       int &ready,
                       std::mutex &mtx,
                       std::condition_variable &cv,
                       MPI_Comm comm) {
    PE(multiply_communication_other);

    int local_size = m.length() * n.subinterval(divisor, gp).length();
    MPI_Win win = create_window(comm, recv_buffer, local_size, false);
    auto mpi_type = mpi_mapper<Scalar>::getType();

    int packages = 0;
    int i = 0;
    while (packages < divisor) {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [packages, divisor, jump_size, &ready]() {
            int diff = ready - packages;
            return diff >= jump_size || (divisor - packages < jump_size);
        });

        packages = ready;
        lk.unlock();
        packages = std::min(packages, divisor);

        int diff = packages - i;
        while (i < packages) {
            int idx = (gp + i) % divisor;
            Scalar *pointer_c =
                expanded_matrix + m.length() * displacements[idx];
            int b_size = m.length() * n.subinterval(divisor, idx).length();

            PL();
            PE(multiply_communication_reduce);
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, idx, 0, win);
            MPI_Accumulate(pointer_c,
                           b_size,
                           mpi_type,
                           idx,
                           0,
                           b_size,
                           mpi_type,
                           MPI_SUM,
                           win);
            MPI_Win_unlock(idx, win);
            PL();
            PE(multiply_communication_other);
            i++;
        }
    }

    MPI_Win_free(&win);
    PL();
}

template <typename Scalar>
void compute(cosma_context<Scalar> *ctx,
             CosmaMatrix<Scalar> &A,
             CosmaMatrix<Scalar> &B,
             CosmaMatrix<Scalar> &C,
             Scalar *pointer_b,
             Scalar *pointer_c,
             Interval &m,
             Interval &n,
             Interval &k,
             std::vector<int> &displacements_n,
             Scalar alpha,
             Scalar beta,
             int start,
             int end) {
    if (start >= end)
        return;

    int n_length = 0;
    if (end >= displacements_n.size()) {
        n_length = n.length() - displacements_n[start];
    } else {
        n_length = displacements_n[end] - displacements_n[start];
    }

    pointer_b += k.length() * displacements_n[start];
    pointer_c += m.length() * displacements_n[start];
    // Scalar* b = pointer_b + k.length() * displacements_n[i];
    // Scalar* c = pointer_c + m.length() * displacements_n[i];

    B.set_current_matrix(pointer_b);
    C.set_current_matrix(pointer_c);
    // B.set_current_matrix(b);
    // C.set_current_matrix(c);

    PL();
    bool copy_c_back = true;
    local_multiply(ctx,
                   A.current_matrix(),
                   B.current_matrix(),
                   C.current_matrix(),
                   m.length(),
                   n_length,
                   k.length(),
                   alpha,
                   beta,
                   copy_c_back);
    PE(multiply_communication_other);
}

// ***********************************
//           DIVISION BY K
// ***********************************
template <typename Scalar>
void overlap_k_split(cosma_context<Scalar> *ctx,
                     MPI_Comm comm,
                     int rank,
                     int divisor,
                     CosmaMatrix<Scalar> &matrixA,
                     CosmaMatrix<Scalar> &matrixB,
                     CosmaMatrix<Scalar> &matrixC,
                     Interval &m,
                     Interval &n,
                     Interval &k,
                     Interval &P,
                     Scalar alpha,
                     Scalar beta) {
    PE(multiply_communication_other);
    // int divisor = strategy.divisor(step);
    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(divisor, rank);

    CosmaMatrix<Scalar> &expanded_mat = matrixC;
    int buffer_idx = expanded_mat.buffer_index();
    expanded_mat.advance_buffer();

    Scalar *original_matrix = expanded_mat.current_matrix();
    Scalar *expanded_matrix = expanded_mat.buffer_ptr();

    expanded_mat.set_buffer_index(buffer_idx);
    expanded_mat.set_current_matrix(original_matrix);

    int local_size = m.length() * n.subinterval(divisor, gp).length();

    auto accumulate_buffer = (beta != Scalar{0})
                                 ? expanded_mat.reduce_buffer_ptr()
                                 : original_matrix;
    std::fill(accumulate_buffer, accumulate_buffer + local_size, Scalar{0});

    Interval newk = k.subinterval(divisor, gp);

    std::vector<int> displacements_n(divisor);
    int disp = 0;
    for (int rank = 0; rank < divisor; ++rank) {
        displacements_n[rank] = disp;
        disp += n.subinterval(divisor, rank).length();
    }
    // c: m * displacements_n
    // b: newk * displacements_n

    // std::atomic_int ready(0);
    int ready = 0;
    std::mutex mtx;
    std::condition_variable cv;

    int comp_comm_ratio = 1;
    int target_jump_size = std::min(comp_comm_ratio, divisor);

    std::thread comm_task(comm_task_k_split<Scalar>,
                          divisor,
                          gp,
                          off,
                          target_jump_size,
                          expanded_matrix,
                          accumulate_buffer,
                          m,
                          n,
                          P,
                          std::ref(displacements_n),
                          std::ref(ready),
                          std::ref(mtx),
                          std::ref(cv),
                          comm);

    Scalar *prev_a = matrixA.current_matrix();
    Scalar *prev_b = matrixB.current_matrix();
    Scalar *prev_c = expanded_matrix;

    int remainder_packages = 0;

    if (target_jump_size == divisor) {
        compute(ctx,
                matrixA,
                matrixB,
                matrixC,
                prev_b,
                prev_c,
                m,
                n,
                newk,
                std::ref(displacements_n),
                alpha,
                Scalar{0},
                0,
                divisor);

        std::unique_lock<std::mutex> lk(mtx);
        ready = divisor;
        lk.unlock();
        cv.notify_one();
    } else {
        int processed = 0;
        int start = gp;
        int end = gp;
        while (processed < divisor) {
            int jump_size = target_jump_size - remainder_packages;
            remainder_packages = 0;
            end = (start + jump_size) % divisor;

            if (start < end) {
                if (start < gp) {
                    end = std::min(end, gp);
                }

                compute(ctx,
                        matrixA,
                        matrixB,
                        matrixC,
                        prev_b,
                        prev_c,
                        m,
                        n,
                        newk,
                        std::ref(displacements_n),
                        alpha,
                        Scalar{0},
                        start,
                        end);

                processed += end - start;
                std::unique_lock<std::mutex> lk(mtx);
                ready += end - start;
                lk.unlock();
                cv.notify_one();

                if (processed < divisor) {
                    int next_end = end + 1;
                    if (next_end <= divisor) {
                        compute(ctx,
                                matrixA,
                                matrixB,
                                matrixC,
                                prev_b,
                                prev_c,
                                m,
                                n,
                                newk,
                                std::ref(displacements_n),
                                alpha,
                                Scalar{0},
                                end,
                                next_end);
                        processed++;
                        remainder_packages = 1;

                        std::unique_lock<std::mutex> lk(mtx);
                        ready++;
                        lk.unlock();
                        cv.notify_one();
                    }
                }
            } else {
                if (end >= gp) {
                    end = std::min(end, gp);
                }

                compute(ctx,
                        matrixA,
                        matrixB,
                        matrixC,
                        prev_b,
                        prev_c,
                        m,
                        n,
                        newk,
                        std::ref(displacements_n),
                        alpha,
                        Scalar{0},
                        start,
                        divisor);
                compute(ctx,
                        matrixA,
                        matrixB,
                        matrixC,
                        prev_b,
                        prev_c,
                        m,
                        n,
                        newk,
                        std::ref(displacements_n),
                        alpha,
                        Scalar{0},
                        0,
                        end);

                processed += divisor - start + end;

                std::unique_lock<std::mutex> lk(mtx);
                ready += divisor - start + end;
                lk.unlock();

                cv.notify_one();

                if (processed < divisor) {
                    int next_end = end + 1;
                    if (next_end <= gp) {
                        compute(ctx,
                                matrixA,
                                matrixB,
                                matrixC,
                                prev_b,
                                prev_c,
                                m,
                                n,
                                newk,
                                std::ref(displacements_n),
                                alpha,
                                Scalar{0},
                                end,
                                next_end);
                        processed++;
                        remainder_packages = 1;

                        std::unique_lock<std::mutex> lk(mtx);
                        ready++;
                        lk.unlock();
                        cv.notify_one();
                    }
                }
            }
            start = (1 + end) % divisor; // t = (end + 1) % divisor;
            // start = (end) % divisor;
        }
        if (remainder_packages > 0) {
            cv.notify_one();
        }
    }

    comm_task.join();

    if (beta != Scalar{0}) {
        for (unsigned i = 0u; i < local_size; ++i) {
            original_matrix[i] =
                original_matrix[i] * beta + accumulate_buffer[i];
        }
    }

    PL();
}

template <typename Scalar>
void overlap_comm_and_comp(cosma_context<Scalar> *ctx,
                           MPI_Comm comm,
                           int rank,
                           const Strategy strategy,
                           CosmaMatrix<Scalar> &matrixA,
                           CosmaMatrix<Scalar> &matrixB,
                           CosmaMatrix<Scalar> &matrixC,
                           Interval &m,
                           Interval &n,
                           Interval &k,
                           Interval &P,
                           size_t step,
                           Scalar alpha,
                           Scalar beta) {
    bool use_busy_waiting = strategy.use_busy_waiting;
    int divisor = strategy.divisor(step);
    if (strategy.split_m(step)) {
        overlap_m_split(use_busy_waiting,
                        ctx,
                        comm,
                        rank,
                        divisor,
                        matrixA,
                        matrixB,
                        matrixC,
                        m,
                        n,
                        k,
                        P,
                        alpha,
                        beta);
    } else if (strategy.split_n(step)) {
        overlap_n_split(use_busy_waiting,
                        ctx,
                        comm,
                        rank,
                        divisor,
                        matrixA,
                        matrixB,
                        matrixC,
                        m,
                        n,
                        k,
                        P,
                        alpha,
                        beta);
    } else {
        overlap_k_split(ctx,
                        comm,
                        rank,
                        divisor,
                        matrixA,
                        matrixB,
                        matrixC,
                        m,
                        n,
                        k,
                        P,
                        alpha,
                        beta);
    }
}

template void overlap_comm_and_comp<float>(cosma_context<float> *ctx,
                                           MPI_Comm comm,
                                           int rank,
                                           const Strategy strategy,
                                           CosmaMatrix<float> &matrixA,
                                           CosmaMatrix<float> &matrixB,
                                           CosmaMatrix<float> &matrixC,
                                           Interval &m,
                                           Interval &n,
                                           Interval &k,
                                           Interval &P,
                                           size_t step,
                                           float alpha,
                                           float beta);

template void overlap_comm_and_comp<double>(cosma_context<double> *ctx,
                                            MPI_Comm comm,
                                            int rank,
                                            const Strategy strategy,
                                            CosmaMatrix<double> &matrixA,
                                            CosmaMatrix<double> &matrixB,
                                            CosmaMatrix<double> &matrixC,
                                            Interval &m,
                                            Interval &n,
                                            Interval &k,
                                            Interval &P,
                                            size_t step,
                                            double alpha,
                                            double beta);

template void overlap_comm_and_comp<std::complex<float>>(
    cosma_context<std::complex<float>> *ctx,
    MPI_Comm comm,
    int rank,
    const Strategy strategy,
    CosmaMatrix<std::complex<float>> &matrixA,
    CosmaMatrix<std::complex<float>> &matrixB,
    CosmaMatrix<std::complex<float>> &matrixC,
    Interval &m,
    Interval &n,
    Interval &k,
    Interval &P,
    size_t step,
    std::complex<float> alpha,
    std::complex<float> beta);

template void overlap_comm_and_comp<std::complex<double>>(
    cosma_context<std::complex<double>> *ctx,
    MPI_Comm comm,
    int rank,
    const Strategy strategy,
    CosmaMatrix<std::complex<double>> &matrixA,
    CosmaMatrix<std::complex<double>> &matrixB,
    CosmaMatrix<std::complex<double>> &matrixC,
    Interval &m,
    Interval &n,
    Interval &k,
    Interval &P,
    size_t step,
    std::complex<double> alpha,
    std::complex<double> beta);

template void overlap_comm_and_comp<bfloat16>(cosma_context<bfloat16> *ctx,
                                              MPI_Comm comm,
                                              int rank,
                                              const Strategy strategy,
                                              CosmaMatrix<bfloat16> &matrixA,
                                              CosmaMatrix<bfloat16> &matrixB,
                                              CosmaMatrix<bfloat16> &matrixC,
                                              Interval &m,
                                              Interval &n,
                                              Interval &k,
                                              Interval &P,
                                              size_t step,
                                              bfloat16 alpha,
                                              bfloat16 beta);

} // end namespace one_sided_communicator

} // namespace cosma
