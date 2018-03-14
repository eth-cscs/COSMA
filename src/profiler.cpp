#include "profiler.h"

//~ Static veriable instrantiations
profiler::time_map profiler::times = profiler::time_map();

profiler::time_diff_map profiler::sum_time = profiler::time_diff_map();

profiler::tree profiler::timer_tree = profiler::tree();

std::string profiler::root = "";

void profiler::print_node(std::string name, int indent) {
    time_map_it it = times.find(name);
    if(it == times.end()) {
        std::cout << "Invalid timer id : " << name << "\n";
    }
    else {
        std::string s = std::string(indent, ' ');
        std::cout << s << "|- " << name << ": " <<
        sum_time[name] << "\n";
    }
}

void profiler::print(std::string node, int indent) {
    print_node(node, indent);
    for(std::set<std::string>::iterator it = timer_tree[node].begin() ; it !=
        timer_tree[node].end() ; it++) {
        print(*it,indent+4);
    }
}

void profiler::print(void) {
    std::cout << "PROFILING RESULTS:\n";
    print(root, -4);
}

void profiler::start(std::string name, std::string father) {
    time_map_it it = times.find(name);
    if(it == times.end()) {
        timer_tree.insert(std::pair<std::string, std::set<std::string> >
                (name,std::set<std::string>()));
        auto now = std::chrono::high_resolution_clock::now();
        times.insert(node(name,now));
        sum_time.insert(node_delta(name,0));
        timer_tree[father].insert(name);
    }
    else {
        auto now = std::chrono::high_resolution_clock::now();
        it->second = now;
    }
}

void profiler::tic(std::string name) {
    time_map_it it = times.find(name);
    if(it == times.end()) {
        std::cout << "Invalid timer id : " << name << "\n";
    }
    else {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_diff = now - it->second;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_diff).count();
        times[name] = now;
        sum_time[name] += elapsed;
    }
}

#ifdef CARMA_HAVE_PROFILING
void profiler_start(std::string name, std::string father) {
    profiler::start(name, father);
}
void profiler_tic(std::string name) {
    profiler::tic(name);
}
void profiler_print() {
    profiler::print();
}
#else
void profiler_start(std::string name, std::string father) {}
void profiler_tic(std::string name) {}
void profiler_print() {}
#endif
