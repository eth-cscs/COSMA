#include "profiler.h"

// Static veriable instrantiations
const std::string profiler::root = "total";
profiler::time_map profiler::times = profiler::time_map();
profiler::time_diff_map profiler::sum_time = profiler::time_diff_map();
profiler::tree profiler::children_tree = profiler::tree();
profiler::f_tree profiler::father_tree = profiler::f_tree();
profiler::counter_map profiler::count = profiler::counter_map();
profiler::counter_map profiler::nested = profiler::counter_map();

float profiler::percentage(std::string node) {
    if (node == root)
        return 100.0f;

    time_type t_father = time(sum_time[father_tree[node]]);
    if (t_father > 0) {
        time_type t_child = time(sum_time[node]);
        return 100.0f * t_child / t_father;
    }

    return 0.0f;
}

profiler::time_type profiler::elapsed_time(profiler::duration_type duration) {
    return std::chrono::duration_cast<time_precision_type>(duration).count();
}

profiler::time_point profiler::current_time() {
    return std::chrono::high_resolution_clock::now();
}

profiler::time_type profiler::time(time_type t) {
    return t;
}

int profiler::counter(std::string node) {
    if (children_tree.find(node) == children_tree.end())
        return -1;
    if (children_tree[node].size() > 0)
        return -1;
    return count[node];
}

void profiler::print_node(std::string name, int indent) {
    time_map_it it = times.find(name);
    if(it == times.end()) {
        std::cout << "Invalid timer id : " << name << "\n";
    }
    else {
        std::string indent_str = std::string(indent, ' ');
        std::string label = indent_str + "|-" + name;
        time_type t = time(sum_time[name]);
        int c = counter(name);
        float percent = percentage(name);

        if (c != -1) {
            printf("%-30s%10.3f%10.1f%10.1d\n",
                        label.c_str(),
                        float(t),
                        float(percent),
                        int(c));
        } else {
            char empty = '-';
            printf("%-30s%10.3f%10.1f%10c\n",
                        label.c_str(),
                        float(t),
                        float(percent),
                        empty);

        }
    }
}

void profiler::print(std::string node, int indent) {
    print_node(node, indent);
    for(std::set<std::string>::iterator it = children_tree[node].begin() ; it !=
        children_tree[node].end() ; it++) {
        print(*it,indent+4);
    }
}

void profiler::print_header(void) {
    std::cout << " -------------------------------------------------------------- \n";
    std::cout << "|                           PROFILER                           |\n";
    std::cout << " -------------------------------------------------------------- \n";
    std::cout << "| region                          t [ms]       [%]       count |\n";
    std::cout << " -------------------------------------------------------------- \n";
}

void profiler::print(void) {
    long long total_time = 0;
    for(std::set<std::string>::iterator it = children_tree[root].begin() ; it !=
        children_tree[root].end() ; it++) {
        total_time += sum_time[*it];
    }
    sum_time[root] = total_time;
    print_header();
    print(root, 0);
}

void profiler::start(std::string name, std::string father) {
    auto nested_it = nested.find(name);
    if (nested_it->second > 0) {
        nested_it->second++;
        return;
    }
    auto now = current_time();
    // insert the root in the tree if not already there
    if (father == root && times.find(father) == times.end()) {
        children_tree.insert(std::pair<std::string, std::set<std::string> >
                (name,std::set<std::string>()));
        times.insert(node(father,now));
        sum_time.insert(node_delta(father,0));
        nested[father] = 1;
    }
    // insert the child in the tree if not already there
    // and update the times
    time_map_it it = times.find(name);
    if(it == times.end()) {
        children_tree.insert(std::pair<std::string, std::set<std::string> >
                (name,std::set<std::string>()));
        times.insert(node(name,now));
        sum_time.insert(node_delta(name,0));
        children_tree[father].insert(name);
        father_tree[name] = father;
        count[name] = 1;
        nested[name] = 1;
    }
    else {
        it->second = now;
        nested_it->second++;
        count[name]++;
    }
}

void profiler::tic(std::string name) {
    time_map_it it = times.find(name);
    if(it == times.end()) {
        std::cout << "Invalid timer id : " << name << "\n";
    }
    else {
        nested[name]--;
        if (nested[name] == 0) {
            auto now = current_time();
            auto elapsed_diff = now - it->second;
            auto elapsed = elapsed_time(now - it->second);
            times[name] = now;
            sum_time[name] += elapsed;
        }
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
