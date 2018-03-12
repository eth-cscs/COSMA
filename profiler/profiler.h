#ifndef _PROFILERH_
#define _PROFILERH_

#include <iostream>
#include <cstdlib>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <sys/time.h>
#include <chrono>

struct profiler {
    typedef std::chrono::system_clock::time_point time_point;
    // tree
    typedef std::map<std::string, time_point> time_map;
    typedef std::map<std::string, long long> time_diff_map;
    typedef time_map::iterator time_map_it;
    // contains node name and absolute time in ms
    typedef std::pair<std::string,time_point> node;
    // contains node name and accumulated relative time diff in ms
    typedef std::pair<std::string,long long> node_delta;
    typedef std::map<std::string, std::set<std::string>> tree;

    // time of each node
    static time_map times;
    // sum of each timer per node
    static time_diff_map sum_time;

    // for each node a set of children
    static tree timer_tree;

    // name of the root node
    static std::string root;

    // private constructor to avoid object creation.
    // we use static funciton calls 
    profiler(void){}

    // starts a timer with "name" id. 
    static void start(
            std::string name, 
            std::string father = ""
    );

    // adds the lapsed time for the node "name" since the last start of that node
    static void tic(std::string name);

    // prints the current node
    static void print_node(std::string name, int indent);
    // invokes print_node for the current node and recurrsively print for all children
    static void print(std::string node, int indent);
    // prints the whole tree - just initiates the call to print_node
    static void print(void);
};

void profiler_start(std::string name, std::string father="");
void profiler_tic(std::string name);
void profiler_print();

// define some helper macros to make instrumentation of the source code with calls
// to the profiler a little less visually distracting
#define PE profiler_start
#define PL profiler_tic
#define PP profiler_print

#endif
