#pragma once
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

// A class describing the interval [start, end)
namespace costa {
struct interval {
    int start = 0;
    int end = 0;

    interval() = default;

    interval(int start, int end);

    int length() const;

    // an interval contains
    bool contains(interval other) const;
    bool contains(int index) const;

    bool non_empty() const;
    bool empty() const;

    interval intersection(const interval &other) const;

    /*
        finds intervals from v that overlap with [start, end),
        i.e. finds start_index and end_index
        (where 0 <= start_index < end_index < v.size())
        that satisfy the following:
          * start_index = max i such that v[i] <= start
          * end_index = min i such that v[i] >= end
     */
    std::pair<int, int> overlapping_intervals(const std::vector<int> &v) const;

    bool operator==(const interval &other) const;
    bool operator!=(const interval &other) const;
    bool operator<(const interval &other) const;
};

std::ostream &operator<<(std::ostream &os, const interval &other);
} // namespace costa
