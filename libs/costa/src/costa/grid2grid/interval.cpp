#include <costa/grid2grid/interval.hpp>

namespace costa {
// A class describing the interval [start, end)
interval::interval(int start, int end)
    : start(start)
    , end(end) {
    if (start < 0 || end < 0) {
        throw std::runtime_error(
            "ERROR: in class interval (costa): start, end > 0 must be satisfied.");
    }

    if (start > end) {
        throw std::runtime_error(
            "ERROR: in class interval (costa): start<=end must be satisfied.");
    }
}

int interval::length() const { return end - start; }

// an interval contains
bool interval::contains(interval other) const {
    return start <= other.start && end >= other.end;
}
//
// an interval contains an index
bool interval::contains(int index) const {
    return start <= index && index < end;
}

bool interval::non_empty() const { return end > start; }

bool interval::empty() const { return end == start; }

interval interval::intersection(const interval &other) const {
    if (!non_empty() || !other.non_empty() || start >= other.end ||
        end <= other.start) {
        return {};
    }
    return {std::max(start, other.start), std::min(end, other.end)};
}

bool interval::operator==(const interval &other) const {
    if (empty())
        return other.empty();
    return start == other.start && end == other.end;
}

bool interval::operator!=(const interval &other) const {
    return !(*this == other);
}

bool interval::operator<(const interval &other) const {
    return start < other.start || 
           (start == other.start && end < other.end);
}

/*
finds intervals from v that overlap with [start, end),
i.e. finds start_index and end_index
(where 0 <= start_index < end_index < v.size())
that satisfy the following:
  * start_index = max i such that v[i] <= start
  * end_index = min i such that v[i] >= end
*/
// TODO: use binary search instead of linear search for this
std::pair<int, int>
interval::overlapping_intervals(const std::vector<int> &v) const {
    if (start >= end || start >= v.back() || end <= v.front())
        return {-1, -1};

    int start_index = 0;
    int end_index = 0;

    for (unsigned i = 0; i < v.size(); ++i) {
        if (v[i] <= start) {
            start_index = i;
        }
        if (v[i] >= end) {
            end_index = i;
            break;
        }
    }
    if (v[start_index] <= start && v[start_index + 1] > start &&
        v[end_index] >= end && v[end_index - 1] < end)
        return {start_index, end_index};
    else
        throw std::runtime_error("bug in overlapping intervals function.");
}

std::ostream &operator<<(std::ostream &os, const interval &other) {
    return os << "interval[" << other.start << ", " << other.end << ")"
              << std::endl;
}
} // namespace costa
