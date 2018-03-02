#ifndef _INTERVALH_
#define _INTERVALH_

#include <iostream>
#include <vector>

// interval of consecutive numbers
struct Interval {
    int start_;
    int end_;

    Interval() {}

    Interval(int start, int end) : start_(start), end_(end) {}

    const int first() {
        return start_;
    }

    const int last() {
        return end_;
    }

    int length() {
        return end_ - start_ + 1;
    }

    bool empty() {
        return start_ == end_;
    }

    bool only_one() {
        return length() == 1;
    }

    // divides the interval into intervals of equal length.
    // if the interval is not divisible by divisor, then
    // last interval might not be of the same size as others.
    std::vector<Interval> divide_by(int divisor) {
        if (length() < divisor) {
            return {*this};
        }

        std::vector<Interval> divided(divisor);

        for (int i = 0; i < divisor; i++) {
            divided[i] = subinterval(divisor, i);
        }

        return divided;
    }

    int partition_index(int divisor, int elem) {
        return (elem-first()) / (length() / divisor);
    }

    // returns the interval containing elem
    Interval subinterval_containing(int divisor, int elem) {
        return subinterval(divisor, partition_index(divisor, elem));
    }

    // returns the box_index-th interval
    Interval subinterval(int divisor, int box_index) {
        if (length() < divisor) {
            return {*this};
        }

        // this will interleave smaller and bigger intervals
        int start = length() * box_index / divisor;
        int end = length() * (box_index + 1) / divisor - 1;

        // alternative that will first have all the bigger intervals and then the smaller ones
        // int interval_length = length() / divisor + (box_index <= length() % divisor ? 1 : 0)

        return Interval(start_ + start, start_ + end);
    }

    friend std::ostream& operator<<(std::ostream& os, const Interval& inter) {
        os << '[' << inter.start_ << ", " << inter.end_ << ']';
        return os;
    }

    bool contains(int num) {
        return num >= first() && num <= last();
    }

    bool contains(Interval other) {
        return first() <= other.first() && last() >= other.last();
    }

    bool before(Interval& other) {
        return last() < other.first();
    }

    bool operator==(const Interval &other) const {
        return start_ == other.start_ && end_ == other.end_;
    }
};

struct Interval2D {
    Interval rows;
    Interval cols;

    Interval2D(Interval row, Interval col) : rows(row), cols(col) {}
    Interval2D(int row_start, int row_end, int col_start, int col_end) {
        rows = Interval(row_start, row_end);
        cols = Interval(col_start, col_end);
    }

    // splits the current Interval2D into divisor many submatrices by splitting
    // only the columns interval and returns the size of the submatrix indexed with index
    int split_by(int divisor, int index) {
        if (index >= divisor) {
            std::cout << "Error in Interval2D.split_by: trying to access " << index << "-subinterval, out of " << divisor << " total subintervals\n";
            return -1;
        }

        if (cols.length() < divisor) {
            std::cout << "Error in Interval2D.split_by: trying to divide the subinterval of length " << cols.length() << " into " << divisor << " many subintervals\n";
            return -1;
        }

        return rows.length() * cols.subinterval(divisor, index).length();
    }

    int size() {
        int size = split_by(1, 0);
        return size;
    }

    bool contains(int row, int col) {
        return rows.contains(row) && cols.contains(col);
    }

    bool contains(Interval2D other) {
        return rows.contains(other.rows) && cols.contains(other.cols);
    }

    bool before(Interval2D& other) {
        return (rows.before(other.rows) && other.cols.contains(cols))|| (cols.before(other.cols) && other.rows.contains(rows));
      //return (rows.before(other.rows))|| (cols.before(other.cols));
    }

    int local_index(int row, int col) {
        if (!contains(row, col)) {
            return -1;
        }
        row -= rows.first();
        col -= cols.first();
        return col * rows.length() + row;
    }



    std::pair<int, int> global_index(int local_index) {
        int x, y;
        x = rows.first() + local_index % rows.length();
        y = cols.first() + local_index / rows.length();
        return {x, y};
    }

    Interval2D submatrix(int divisor, int index) {
        return Interval2D(rows, cols.subinterval(divisor, index));
    }

    bool operator==(const Interval2D &other) const {
        return (rows == other.rows) && (cols == other.cols);
    }

    friend std::ostream& operator<<(std::ostream& os, const Interval2D& inter) {
        os << "rows " << inter.rows << "; columns: " << inter.cols;
        return os;
    }
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
