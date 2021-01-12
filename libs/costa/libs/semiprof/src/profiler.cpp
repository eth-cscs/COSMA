#include <algorithm>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <mutex>
#include <ostream>
#include <thread>

#include <semiprof/semiprof.hpp>

namespace semiprof {

using clock_type = std::chrono::high_resolution_clock;
using time_point_type = clock_type::time_point;
using duration_type = time_point_type::duration;

#ifdef SEMIPROF
namespace {
    unsigned n_threads = 100;

    struct thread_info {
        unsigned thread_id;
        unsigned num_threads;
    };

    inline
    thread_info get_thread_info() {
        static std::atomic<unsigned> num_threads(0);
        thread_local unsigned thread_id = num_threads++;
        return {thread_id, num_threads};
    };

    // Check whether a string describes a valid profiler region name.
    bool is_valid_region_string(const std::string& s) {
        if (s.size()==0u || s.front()=='_' || s.back()=='_') return false;
        return s.find("__") == s.npos;
    }

    //
    // Return a list of the words in the string, using '_' as the delimiter
    // string, e.g.:
    //      "communicator"             -> {"communicator"}
    //      "communicator_events"      -> {"communicator", "events"}
    //      "communicator_events_sort" -> {"communicator", "events", "sort"}
    std::vector<std::string> split(const std::string& str) {
        std::vector<std::string> cont;
        std::size_t first = 0;
        std::size_t last = str.find('_');
        while (last != std::string::npos) {
            cont.push_back(str.substr(first, last - first));
            first = last + 1;
            last = str.find('_', first);
        }
        cont.push_back(str.substr(first, last - first));
        return cont;
    }
}

// Holds the accumulated number of calls and time spent in a region.
struct profile_accumulator {
    std::size_t count;
    duration_type time;

    profile_accumulator() = default;
};

// Records the accumulated time spent in profiler regions on one thread.
// There is one recorder for each thread.
class recorder {
    // used to mark that the recorder is not currently timing a region.
    static constexpr region_id_type npos = std::numeric_limits<region_id_type>::max();

    // The index of the region being timed.
    // If set to npos, no region is being timed.
    region_id_type index_ = npos;

    time_point_type start_time_;

    // One accumulator for call count and wall time for each region.
    std::vector<profile_accumulator> accumulators_;

public:
    // Return a list of the accumulated call count and wall times for each region.
    const std::vector<profile_accumulator>& accumulators() const;

    // Start timing the region with index.
    // Throws std::runtime_error if already timing a region.
    void enter(region_id_type index);

    // Stop timing the current region, and add the time taken to the accumulated time.
    // Throws std::runtime_error if not currently timing a region.
    void leave();

    // Reset all of the accumulated call counts and times to zero.
    void clear();
};

// Manages the thread-local recorders.
class profiler {
    std::vector<recorder> recorders_;

    // Hash table that maps region names to a unique index.
    // The regions are assigned consecutive indexes in the order that they are
    // added to the profiler with calls to `region_index()`, with the first
    // region numbered zero.
    std::unordered_map<const char*, region_id_type> name_index_;

    // The name of each region being recorded, with index stored in name_index_
    // is used to index into region_names_.
    std::vector<std::string> region_names_;

    // Used to protect name_index_, which is shared between all threads.
    std::mutex mutex_;

public:
    profiler(unsigned max_threads=256);

    void clear();
    void enter(region_id_type index);
    void enter(const char* name);
    void leave();
    const std::vector<std::string>& regions() const;
    region_id_type region_index(const char* name);
    profile results() const;

    static profiler& get_global_profiler() {
        static profiler p;
        return p;
    }
};


// Utility structure used to organise profiler regions into a tree for printing.
struct profile_node {
    static constexpr region_id_type npos = std::numeric_limits<region_id_type>::max();

    std::string name;
    double time = 0;
    region_id_type count = npos;
    std::vector<profile_node> children;

    profile_node() = default;
    profile_node(std::string n, double t, region_id_type c):
        name(std::move(n)), time(t), count(c) {}
    profile_node(std::string n):
        name(std::move(n)), time(0), count(npos) {}
};

// recorder implementation

const std::vector<profile_accumulator>& recorder::accumulators() const {
    return accumulators_;
}

void recorder::enter(region_id_type index) {
    if (index_!=npos) {
        throw std::runtime_error("recorder::enter without matching recorder::leave");
    }
    if (index>=accumulators_.size()) {
        accumulators_.resize(index+1);
    }
    index_ = index;
    start_time_ = clock_type::now();
}

void recorder::leave() {
    // calculate the elapsed time before any other steps, to increase accuracy.
    auto delta = clock_type::now() - start_time_;

    if (index_==npos) {
        throw std::runtime_error("recorder::leave without matching recorder::enter");
    }
    accumulators_[index_].count++;
    accumulators_[index_].time += delta;
    index_ = npos;
}

void recorder::clear() {
    index_ = npos;
    accumulators_.resize(0);
}

// profiler implementation

profiler::profiler(unsigned max_threads) {
    recorders_.resize(max_threads);
}

void profiler::enter(region_id_type index) {
    recorders_[get_thread_info().thread_id].enter(index);
}

void profiler::enter(const char* name) {
    const auto index = region_index(name);
    recorders_[get_thread_info().thread_id].enter(index);
}

void profiler::leave() {
    recorders_[get_thread_info().thread_id].leave();
}

void profiler::clear() {
    recorders_[get_thread_info().thread_id].clear();
}

region_id_type profiler::region_index(const char* name) {
    // The name_index_ hash table is shared by all threads, so all access
    // has to be protected by a mutex.
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = name_index_.find(name);
    if (it==name_index_.end()) {
        const auto index = region_names_.size();
        name_index_[name] = index;
        region_names_.emplace_back(name);
        return index;
    }
    return it->second;
}

// Used to prepare the profiler output for printing.
// Perform a depth first traversal of a profile tree that:
// - sorts the children of each node in ascending order of time taken;
// - sets the time taken for each non-leaf node to the sum of its children.
double sort_profile_tree(profile_node& n) {
    // accumulate all time taken in children
    if (!n.children.empty()) {
        n.time = 0;
        for (auto &c: n.children) {
            sort_profile_tree(c);
            n.time += c.time;
        }
    }

    // sort the children in descending order of time taken
    //util::sort_by(n.children, [](const profile_node& n){return -n.time;});
    std::sort(
        n.children.begin(), n.children.end(),
        [](const profile_node& l, const profile_node& r){return r.time<l.time;});

    return n.time;
}

profile profiler::results() const {
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;

    const auto nregions = region_names_.size();

    profile p;
    p.names = region_names_;

    p.times = std::vector<double>(nregions);
    p.counts = std::vector<region_id_type>(nregions);
    const auto num_threads = get_thread_info().num_threads;
    for (auto tid=0u; tid<num_threads; ++tid) {
        auto& r = recorders_[tid];
        auto& accumulators = r.accumulators();
        for (auto i=0u; i<accumulators.size(); ++i) {
            // convert counter from tics to nanoseconds
            auto time_in_ns = duration_cast<nanoseconds>(accumulators[i].time).count();
            p.times[i]  += time_in_ns * 1e-9; // convert to seconds
            p.counts[i] += accumulators[i].count;
        }
    }

    p.num_threads = get_thread_info().num_threads;

    return p;
}

profile_node make_profile_tree(const profile& p) {
    // Take the name of each region, and split into a sequence of sub-region-strings.
    // e.g. "advance_integrate_state" -> "advance", "integrate", "state"
    std::vector<std::vector<std::string>> names;
    names.reserve(p.names.size());
    for (auto& n: p.names) {
        names.push_back(split(n));
    }

    // Build a tree description of the regions and sub-regions in the profile.
    profile_node tree("total");
    for (auto idx=0u; idx<p.names.size(); ++idx) {
        profile_node* node = &tree;
        const auto depth  = names[idx].size();
        for (auto i=0u; i<depth-1; ++i) {
            auto& node_name = names[idx][i];
            auto& kids = node->children;

            // Find child of node that matches node_name
            auto child = std::find_if(
                kids.begin(), kids.end(), [&](profile_node& n){return n.name==node_name;});

            if (child==kids.end()) { // Insert an empty node in the tree.
                node->children.emplace_back(node_name);
                node = &node->children.back();
            }
            else { // Node already exists.
                node = &(*child);
            }
        }

        node->children.emplace_back(names[idx].back(), p.times[idx], p.counts[idx]);
    }
    sort_profile_tree(tree);

    return tree;
}

const std::vector<std::string>& profiler::regions() const {
    return region_names_;
}

void print(std::ostream& o,
           profile_node& n,
           float wall_time,
           unsigned nthreads,
           float thresh,
           std::string indent="")
{
    thread_local char buf[80];

    auto name = indent + n.name;
    float per_thread_time = n.time/nthreads;
    float proportion = n.time/wall_time*100;

    // If the percentage of overall time for this region is below the
    // threashold, stop drawing this branch.
    if (proportion<thresh) return;

    if (n.count==profile_node::npos) {
        snprintf(buf, sizeof(buf)/sizeof(char),
                "_p_ %-20s%12s%12.3f%12.3f%8.1f",
                name.c_str(), "-", n.time, per_thread_time, proportion);
    }
    else {
        snprintf(buf, sizeof(buf)/sizeof(char),
                "_p_ %-20s%12lu%12.3f%12.3f%8.1f",
                name.c_str(), n.count, n.time, per_thread_time, proportion);
    }
    o << "\n" << buf;

    // print each of the children in turn
    for (auto& c: n.children) print(o, c, wall_time, nthreads, thresh, indent+"  ");
};

//
// convenience functions for instrumenting code.
//

void profiler_leave() {
    profiler::get_global_profiler().leave();
}

region_id_type profiler_region_id(const char* name) {
    if (!is_valid_region_string(name)) {
        throw std::runtime_error(std::string("'")+name+"' is not a valid profiler region name.");
    }
    return profiler::get_global_profiler().region_index(name);
}

void profiler_enter(region_id_type region_id) {
    profiler::get_global_profiler().enter(region_id);
}

void profiler_clear() {
    profiler::get_global_profiler().clear();
}

// Print profiler statistics to an ostream
std::ostream& operator<<(std::ostream& o, const profile& prof) {
    char buf[80];

    auto tree = make_profile_tree(prof);

    snprintf(buf, sizeof(buf)/sizeof(char),
            "_p_ %-20s%12s%12s%12s%8s",
            "REGION", "CALLS", "THREAD", "WALL", "\%");
    o << buf;
    print(o, tree, tree.time, prof.num_threads, 0, "");
    return o;
}

profile profiler_summary() {
    return profiler::get_global_profiler().results();
}

#else

void profiler_leave() {}
void profiler_enter(region_id_type) {}
void profiler_clear() {}
profile profiler_summary();
profile profiler_summary() {return profile();}
region_id_type profiler_region_id(const char*) {return 0;}
std::ostream& operator<<(std::ostream& o, const profile&) {return o;}

#endif // SEMIPROF

} // namespace semiprof
