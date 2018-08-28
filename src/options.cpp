#include "options.hpp"

// finds the position after the defined flag or throws an exception 
// if flag is not found in the line.
int options::find_flag(const std::string& short_flag, const std::string& long_flag, 
        const std::string& message, const std::string& line, bool compulsory) {
    auto position = line.find(short_flag);
    auto length = short_flag.length();
    if (position == std::string::npos) {
        position = line.find(long_flag);
        length = long_flag.length();
        if (position == std::string::npos) {
            if (compulsory)
                throw std::runtime_error(message);
            else
                return -1;
        }
    }
    while (line[position + length] == ' ') length++;
    return position + length;
}

// looks for the defined flag in the line
// if found return true, otherwise returns false
bool options::flag_exists(const std::string& short_flag, const std::string& long_flag, 
        const std::string& line) {
    auto position = line.find(short_flag);
    if (position == std::string::npos) {
        position = line.find(long_flag);
        if (position == std::string::npos) {
            return false;
        }
    }
    return true;
}

// finds the next int after start in the line
int options::next_int(int start, const std::string& line) {
    if (start < 0) 
        return -1;
    std::regex int_expr("([0-9]+)");
    auto it = std::sregex_iterator(line.begin() + start, line.end(), int_expr);
    int result = std::stoi(it->str());
    return result;
}

// finds the next int after start in the line
long long options::next_long_long(int start, const std::string& line) {
    if (start < 0) 
        return -1;
    std::regex int_expr("([0-9]+)");
    auto it = std::sregex_iterator(line.begin() + start, line.end(), int_expr);
    long long result = std::stoll(it->str());
    return result;
}

