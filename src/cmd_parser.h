#ifndef CMD_PARSER_H
#define CMD_PARSER_H

//STL
#include <cstring>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string>
#include <cstring>

int find_option( int argc, char **argv, char const* option );
int read_int( int argc, char **argv, char const* option, int default_value );
std::string read_string( int argc, char **argv, char const* option, char const* default_value );

#endif 
