#include <string>
#include <regex>
#include <stdexcept>

namespace options {
    // finds the position after the defined flag or throws an exception 
    // if flag is not found in the line.
    int find_flag(const std::string& short_flag, const std::string& long_flag, 
                     const std::string& message, const std::string& line, 
                     bool throw_exception=true);

    // looks for the defined flag in the line
    // if found return true, otherwise returns false
    bool flag_exists(const std::string& short_flag, const std::string& long_flag, 
            const std::string& line);

    // finds the next int after start in the line
    int next_int(int start, const std::string& line);
    long long next_long_long(int start, const std::string& line);
};
