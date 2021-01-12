#pragma once
#include <unordered_map>
#include <cassert>
#include <iostream>

namespace costa {
struct edge_t {
    int src;
    int dest;

    edge_t() = default;
    edge_t(int src, int dest):
        src(src), dest(dest) {}
    // edge_t(edge_t& e): src(e.src), dest(e.dest) {}

    edge_t sorted() const {
        int u = std::min(src, dest);
        int v = std::max(src, dest);
        return edge_t{u, v};
    }
    bool operator==(const edge_t& other) const {
        return src==other.src && dest==other.dest;
    }
};
}

template <class T>
inline void combine_hash(std::size_t &s, const T &v) {
    std::hash<T> h;
    s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

// add hash function specialization for these struct-s
// so that we can use this class as a key of the unordered_map
namespace std {
template <>
struct hash<costa::edge_t> {
    std::size_t operator()(const costa::edge_t &k) const {
        using std::hash;

        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        size_t result = 0;
        combine_hash(result, k.src);
        combine_hash(result, k.dest);
        return result;
    }
};
} // namespace std

namespace costa {
struct weighted_edge_t {
    edge_t e;
    int w;

    weighted_edge_t() = default;
    weighted_edge_t(int src, int dest, int weight):
        e{src, dest}, w(weight) {}

    const int weight() const {
        return w;
    }

    const int src() const {
        return e.src;
    }

    const int dest() const {
        return e.dest;
    }

    const edge_t& edge() const {
        return e;
    }

    bool operator==(const weighted_edge_t& other) const {
        return e==other.edge() && w == other.weight();
    }

    bool operator<(const weighted_edge_t& other) const {
        return w < other.w;
    }
};

struct comm_volume {
    using volume_t = std::unordered_map<edge_t, int>;
    volume_t volume;

    comm_volume() = default;
    comm_volume(volume_t&& v):
        volume(std::forward<volume_t>(v)) {}

    comm_volume& operator+=(const comm_volume& other) {
        for (const auto& vol : other.volume) {
            auto& e = vol.first;
            int w = vol.second;
            volume[e.sorted()] += w;
        }
        return *this;
    }

    comm_volume operator+(const comm_volume& other) const {
      comm_volume res;
      for (const auto& vol : volume) {
          auto& e = vol.first;
          auto w = vol.second;
          res.volume[e.sorted()] += w;
      }
        for (const auto& vol : other.volume) {
            auto& e = vol.first;
            auto w = vol.second;
            res.volume[e.sorted()] += w;
        }
        return res;
    }

    size_t total_volume() {
        size_t sum = 0;
        for (const auto& vol : volume) {
            auto& e = vol.first;
            int w = vol.second;
            // if not a local communication, count it
            if (e.src != e.dest) {
                assert(w > 0);
                sum += (size_t) w;
            }
        }
        return sum;
    }

    friend std::ostream &operator<<(std::ostream &os, const comm_volume &other) {
        os << "Communication volume consists of the following:" << std::endl;
        for (const auto& vol : other.volume) {
            auto& e = vol.first;
            int w = vol.second;
            os << e.src << "->" << e.dest << ": " << w << std::endl;
        }
        return os;
    }
};
}


