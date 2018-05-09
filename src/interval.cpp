#include "interval.hpp"

// interval of consecutive numbers
Interval::Interval() = default;

Interval::Interval(int start, int end) : start_(start), end_(end) {}

const int Interval::first() {
    return start_;
}

const int Interval::last() {
    return end_;
}

int Interval::length() {
    return end_ - start_ + 1;
}

bool Interval::empty() {
    return start_ == end_;
}

bool Interval::only_one() {
    return length() == 1;
}

// divides the interval into intervals of equal length.
// if the interval is not divisible by divisor, then
// last interval might not be of the same size as others.
std::vector<Interval> Interval::divide_by(int divisor) {
    if (length() < divisor) {
        return {*this};
    }

    std::vector<Interval> divided(divisor);

    for (int i = 0; i < divisor; i++) {
        divided[i] = subinterval(divisor, i);
    }

    return divided;
}

int Interval::partition_index(int divisor, int elem) {
    return (elem-first()) / (length() / divisor);
}

// returns the interval containing elem
Interval Interval::subinterval_containing(int divisor, int elem) {
    return subinterval(divisor, partition_index(divisor, elem));
}

// returns the box_index-th interval
Interval Interval::subinterval(int divisor, int box_index) {
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

int Interval::largest_subinterval_length(int divisor) {
    return length() / divisor + (length() % divisor == 0 ? 0 : 1);
}

int Interval::smallest_subinterval_length(int divisor) {
    return length() / divisor;
}

std::ostream& operator<<(std::ostream& os, const Interval& inter) {
    os << '[' << inter.start_ << ", " << inter.end_ << ']';
    return os;
}

bool Interval::contains(int num) {
    return num >= first() && num <= last();
}

bool Interval::contains(Interval other) {
    return first() <= other.first() && last() >= other.last();
}

bool Interval::before(Interval& other) {
    return last() < other.first();
}

bool Interval::operator==(const Interval &other) const {
    return start_ == other.start_ && end_ == other.end_;
}

Interval2D::Interval2D() = default;
Interval2D::Interval2D(Interval row, Interval col) : rows(row), cols(col) {}

Interval2D::Interval2D(int row_start, int row_end, int col_start, int col_end) {
    rows = Interval(row_start, row_end);
    cols = Interval(col_start, col_end);
}

// splits the current Interval2D into divisor many submatrices by splitting
// only the columns interval and returns the size of the submatrix indexed with index
int Interval2D::split_by(int divisor, int index) {
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

int Interval2D::size() {
    int size = split_by(1, 0);
    return size;
}

bool Interval2D::contains(int row, int col) {
    return rows.contains(row) && cols.contains(col);
}

bool Interval2D::contains(Interval2D other) {
    return rows.contains(other.rows) && cols.contains(other.cols);
}

bool Interval2D::before(Interval2D& other) {
    return (rows.before(other.rows) && other.cols.contains(cols))|| (cols.before(other.cols) && other.rows.contains(rows));
    //return (rows.before(other.rows))|| (cols.before(other.cols));
}

int Interval2D::local_index(int row, int col) {
    if (!contains(row, col)) {
        return -1;
    }
    row -= rows.first();
    col -= cols.first();
    return col * rows.length() + row;
}



std::pair<int, int> Interval2D::global_index(int local_index) {
    int x, y;
    x = rows.first() + local_index % rows.length();
    y = cols.first() + local_index / rows.length();
    return {x, y};
}

Interval2D Interval2D::submatrix(int divisor, int index) {
    return Interval2D(rows, cols.subinterval(divisor, index));
}

bool Interval2D::operator==(const Interval2D &other) const {
    return (rows == other.rows) && (cols == other.cols);
}

std::ostream& operator<<(std::ostream& os, const Interval2D& inter) {
    os << "rows " << inter.rows << "; columns: " << inter.cols;
    return os;
}

