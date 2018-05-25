#pragma once
#include "interval.hpp"
#include "strategy.hpp"
#include <vector>
#include "mapper.hpp"
#include "layout.hpp"

class Buffer {
public:
    Buffer() = default;
    Buffer(char label, const Strategy& strategy, int rank, Mapper* mapper, Layout* layout);

    void initialize_buffers();

    void advance_buffer();
    int buffer_index();
    void set_buffer_index(int idx);

    double* buffer_ptr();
    std::vector<double>& buffer();
    const std::vector<double>& buffer() const;

    double* initial_buffer_ptr();
    std::vector<double>& initial_buffer();
    const std::vector<double>& initial_buffer() const;

    std::vector<double>& operator[](const std::vector<double>::size_type index);
    std::vector<double> operator[](const std::vector<double>::size_type index) const;

    char label_;
    const Strategy* strategy_;
    int rank_;

    Mapper* mapper_;
    Layout* layout_;

protected:
    std::vector<long long> compute_buffer_size();

    std::vector<long long> compute_buffer_size(Interval& m, Interval& n, Interval& k, Interval& P, 
        int step, int rank);

    void compute_max_buffer_size(Interval& m, Interval& n, Interval& k, Interval& P, 
        int step, int rank);

    void compute_max_buffer_size();

    void compute_n_buckets();

    /// local send buffer
    std::vector<std::vector<double>> buffers_;
    int current_buffer_;

    long long max_send_buffer_size_;
    long long max_recv_buffer_size_;

    // computes the number of buckets in the current step
    // the number of buckets in some step i is equal to the
    // product of all divisors in DFS steps that follow step i
    // in which the current matrix was divided
    std::vector<int> n_buckets_;
    std::vector<bool> expanded_after_;

    const long long max_send_buffer_size() const;
    const long long max_recv_buffer_size() const;

};
