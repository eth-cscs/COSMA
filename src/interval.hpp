#ifndef _INTERVALH_
#define _INTERVALH_

#include <iostream>
#include <vector>

// interval of consecutive numbers
class Interval {
public:
    int start_;
    int end_;

    Interval();
    Interval(int start, int end);

    const int first();
    const int last();

    int length();
    bool empty();
    bool only_one();

    // divides the interval into intervals of equal length.
    // if the interval is not divisible by divisor, then
    // last interval might not be of the same size as others.
    std::vector<Interval> divide_by(int divisor);

    int partition_index(int divisor, int elem);

    // returns the interval containing elem
    Interval subinterval_containing(int divisor, int elem);

    // returns the box_index-th interval
    Interval subinterval(int divisor, int box_index);

    // returns the largest subinterval when divided by divisor
    int largest_subinterval_length(int divisor);

    // returns the smallest subinterval when divided by divisor
    int smallest_subinterval_length(int divisor);

    bool contains(int num);
    bool contains(Interval other);
    bool before(Interval& other);

    bool operator==(const Interval &other) const;

    friend std::ostream& operator<<(std::ostream& os, const Interval& inter);
};

class Interval2D {
public:
    Interval rows;
    Interval cols;

    Interval2D();
    Interval2D(Interval row, Interval col);
    Interval2D(int row_start, int row_end, int col_start, int col_end);

    // splits the current Interval2D into divisor many submatrices by splitting
    // only the columns interval and returns the size of the submatrix indexed with index
    int split_by(int divisor, int index);

    int size();

    bool contains(int row, int col);
    bool contains(Interval2D other);
    bool before(Interval2D& other);

    int local_index(int row, int col);
    std::pair<int, int> global_index(int local_index);

    Interval2D submatrix(int divisor, int index);

    bool operator==(const Interval2D &other) const;
    friend std::ostream& operator<<(std::ostream& os, const Interval2D& inter);
};

template <class T>
inline void hash_combine(std::size_t & s, const T & v) {
    std::hash<T> h;
    s^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2);
}

// add hash function specialization for these struct-s
// so that we can use this class as a key of the unordered_map
namespace std {
template <>
struct hash<Interval> {
    std::size_t operator()(const Interval& k) const {
        using std::hash;

        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        size_t result = 0;
        hash_combine(result, k.start_);
        hash_combine(result, k.end_);
        return result;
    }
};

template <>
struct hash<Interval2D> {
    std::size_t operator()(const Interval2D& k) const {
        using std::hash;

        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        size_t result = 0;
        hash_combine(result, k.rows);
        hash_combine(result, k.cols);
        return result;
    }
};
} // namespace std
#endif 
