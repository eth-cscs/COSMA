#include "cmd_parser.h"

//STL
#include <cstring>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string>

int find_option(int argc, char **argv, char const* option) {
    for(int i = 1; i < argc; i++)
    if(strcmp(argv[i], option) == 0)
        return i;
    return -1;
}

int read_int(int argc, char **argv, char const* option, int default_value) {
    int iplace = find_option(argc, argv, option);
    if(iplace >= 0 && iplace < argc-1)
        return atoi(argv[iplace+1]);
    return default_value;
}

std::string read_string( int argc, char **argv,
        char const* option, char const* default_value) {
    int iplace = find_option(argc, argv, option);
    if(iplace >= 0 && iplace < argc-1)
        return argv[iplace+1];
    return default_value;
}
