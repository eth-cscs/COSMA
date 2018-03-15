#ifndef _PROFILERH_
#define _PROFILERH_

#include <iostream>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <set>
#include <fstream>
#include <sys/time.h>
#include <chrono>

struct profiler {
    typedef std::chrono::system_clock::time_point time_point;
    // tree
    typedef std::unordered_map<std::string, int> counter_map;
    typedef std::unordered_map<std::string, time_point> time_map;
    typedef std::unordered_map<std::string, long long> time_diff_map;
    typedef time_map::iterator time_map_it;
    // contains node name and absolute time in ms
    typedef std::pair<std::string,time_point> node;
    // contains node name and accumulated relative time diff in ms
    typedef std::pair<std::string,long long> node_delta;
    typedef std::unordered_map<std::string, std::set<std::string>> tree;
    typedef std::chrono::system_clock::duration duration_type;
    typedef long long time_type;
    // we measure and output the time in milliseconds
    typedef std::chrono::milliseconds time_precision_type;
    typedef std::unordered_map<std::string, std::string> f_tree;

    // time of each node
    static time_map times;
    // sum of each timer per node
    static time_diff_map sum_time;
    static counter_map count;
    static counter_map nested;

    // for each node a set of children
    static tree children_tree;
    static f_tree father_tree;

    // name of the root node
    static const std::string root;

    // gets elapsed time between two time points
    static time_type elapsed_time(duration_type duration);
    // gets the current time point
    static time_point current_time();
    // transforms the measured time to the output time
    // only makes sense if we measure the time at one
    // precision but want to output the time in another precision
    // then this acts as a converter
    static time_type time(time_type t);

    // private constructor to avoid object creation.
    // we use static funciton calls 
    profiler(void){}

    // starts a timer with "name" id. 
    static void start(std::string name, std::string father = root);

    // adds the lapsed time for the node "name" since the last start of that node
    static void tic(std::string name);

    // gets the percentage of sum_time that child has
    // relative to its father in the tree
    static float percentage(std::string node);

    // gets how many times this region was entered
    static int counter(std::string node);

    // prints the header of the table
    static void print_header(void);
    // prints the current node
    static void print_node(std::string name, int indent);
    // invokes print_node for the current node and recurrsively print for all children
    static void print(std::string node, int indent);
    // prints the whole tree - just initiates the call to print_node
    static void print(void);
};

void profiler_start(std::string name, std::string father=profiler::root);
void profiler_tic(std::string name);
void profiler_print();

// define some helper macros to make instrumentation of the source code with calls
// to the profiler a little less visually distracting
#define PE profiler_start
#define PL profiler_tic
#define PP profiler_print

#endif
