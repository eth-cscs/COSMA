#include <cosma/layout.hpp>

namespace cosma {
Layout::Layout(char label,
               int m,
               int n,
               size_t P,
               int rank,
               std::vector<std::vector<Interval2D>> rank_to_range)
    : label_(label)
    , m_(m)
    , n_(n)
    , P_(P)
    , rank_(rank)
    , rank_to_range_(rank_to_range) {
    initial_size_ = std::vector<int>(P);
    bucket_size_ = std::vector<std::vector<int>>(P, std::vector<int>());
    pointer_ = std::vector<int>(P);

    for (size_t p = 0; p < P; ++p) {
        int sum = 0;
        auto ranges = rank_to_range[p];

        for (size_t bucket = 0; bucket < ranges.size(); ++bucket) {
            int size = ranges[bucket].size();
            bucket_size_[p].push_back(size);
            sum += size;
        }
        initial_size_[p] = sum;
    }
}

int Layout::size() { return size(rank_); }

int Layout::size(int rank) { return bucket_size_[rank][pointer_[rank]]; }

// we cannot use the precomputed bucket_offset_ here, since
// the buckets might have increased due to communication
// if parallel and sequential steps are interleaved
int Layout::offset(int rank, int prev_pointer) {
    int sum = 0;

    for (int pointer = prev_pointer; pointer < pointer_[rank]; ++pointer) {
        sum += bucket_size_[rank][pointer];
    }

    return sum;
}

int Layout::offset(int prev_pointer) { return offset(rank_, prev_pointer); }

void Layout::next(int rank) { pointer_[rank]++; }

void Layout::next() {
    // move the pointer to the next range that this rank owns
    // and put its size in buffer_size_[rank_]
    next(rank_);
}

void Layout::prev(int rank) { pointer_[rank]--; }

void Layout::prev() { prev(rank_); }

std::vector<int> Layout::seq_buckets(Interval &newP) {
    std::vector<int> result(newP.length());
    for (int i = newP.first(); i <= newP.last(); ++i) {
        result[i - newP.first()] = pointer_[i];
    }
    return result;
}

void Layout::set_seq_buckets(Interval &newP, std::vector<int> &pointers) {
    for (int i = newP.first(); i <= newP.last(); ++i) {
        pointer_[i] = pointers[i - newP.first()];
    }
}

int Layout::seq_bucket(int rank) { return pointer_[rank]; }

int Layout::seq_bucket() { return seq_bucket(rank_); }

void Layout::update_buckets(Interval &P, Interval2D &range) {
    for (int rank = P.first(); rank <= P.last(); ++rank) {
        int pointer = pointer_[rank];
        auto &ranges = rank_to_range_[rank];

        while (pointer < ranges.size() && ranges[pointer].before(range)) {
            next(rank);
            pointer++;
        }
    }
}

void Layout::buffers_before_expansion(
    Interval &P,
    Interval2D &range,
    std::vector<std::vector<int>> &size_per_rank,
    std::vector<int> &total_size_per_rank) {

    for (int i = P.first(); i <= P.last(); ++i) {
        size_per_rank[i - P.first()] =
            sizes_inside_range(range, i, total_size_per_rank[i - P.first()]);
    }
}

void Layout::buffers_after_expansion(
    Interval &P,
    Interval &newP,
    std::vector<std::vector<int>> &size_per_rank,
    std::vector<int> &total_size_per_rank,
    std::vector<std::vector<int>> &new_size,
    std::vector<int> &new_total) {
    int subset_size = newP.length();
    int div = P.length() / newP.length();

    for (int comm_ring = 0; comm_ring < newP.length(); ++comm_ring) {
        int n_bucket = size_per_rank[comm_ring].size();
        new_size[comm_ring] = std::vector<int>(n_bucket);

        for (int bucket = 0; bucket < n_bucket; ++bucket) {
            for (int group = 0; group < div; ++group) {
                int rank = group * subset_size + comm_ring;
                new_size[comm_ring][bucket] += size_per_rank[rank][bucket];
            }
            new_total[comm_ring] += new_size[comm_ring][bucket];
        }
    }
}

void Layout::set_sizes(Interval &newP,
                       std::vector<std::vector<int>> &size_per_rank,
                       int offset) {
    for (int i = newP.first(); i <= newP.last(); ++i) {
        set_sizes(i, size_per_rank[i - newP.first() + offset], pointer_[i]);
    }
}

void Layout::set_sizes(Interval &newP,
                       std::vector<std::vector<int>> &size_per_rank) {
    for (int i = newP.first(); i <= newP.last(); ++i) {
        set_sizes(i, size_per_rank[i - newP.first()], pointer_[i]);
    }
}

void Layout::set_sizes(int rank, std::vector<int> &sizes, int start) {
    int pointer = start;
    auto &b_sizes = bucket_size_[rank];

    for (int i = pointer; i < std::min(sizes.size() + pointer, b_sizes.size());
         ++i) {
        b_sizes[i] = sizes[i - pointer];
    }
}

// get sizes of all ranges inside range of rank and remember the total_size
std::vector<int>
Layout::sizes_inside_range(Interval2D &range, int rank, int &total_size) {
    std::vector<int> sizes;
    total_size = 0;
    int pointer = pointer_[rank];
    auto &ranges = rank_to_range_[rank];
    auto &b_size = bucket_size_[rank];

    while (pointer < ranges.size()) {
        auto &current_range = ranges[pointer];

        if (!range.contains(current_range)) {
            break;
        }

        int current_size = b_size[pointer];

        sizes.push_back(current_size);
        total_size += current_size;

        pointer++;
    }
    return sizes;
}
} // namespace cosma
