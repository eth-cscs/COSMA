/* Simulates the algorithm (without actually computing the matrix multiplication)
 * in order to get the total volume of the communication, the maximum volume of computation
 * done in a single branch and the maximum required buffer size that the algorithm requires.
 */

// STL
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

// CMD parser
#include <cmd_parser.h>

// Local
#include <statistics.hpp>

int main( int argc, char **argv ) {
    auto P = read_int(argc, argv, "-P", 4);
    auto m = read_int(argc, argv, "-m", 4096);
    auto n = read_int(argc, argv, "-n", 4096);
    auto k = read_int(argc, argv, "-k", 4096);
    auto r = read_int(argc, argv, "-r", 4);
    auto char* patt = read_string(argc, argv, "-p", "bbbb");
    std::string pattern(patt);
    auto char* divPatternStr = read_string(argc, argv, "-d", "211211211211");
    std::string divPatternString(divPatternStr);

    //use only lower case
    std::transform(pattern.begin(), pattern.end(), pattern.begin(),
        [](char c) { return std::tolower(c); });
    if ( pattern.size() != r || !std::all_of(pattern.cbegin(),pattern.cend(),
        [](char c) {
          return (c=='b')||(c=='d');
         })) {
           std::cout << "Recursive pattern " << pattern << " malformed expression\n";
           exit(-1);
    }

    std::vector<int> divPattern;
    auto it = divPatternString.cbegin();

    for (int i=0; i<r; i++) {
        bool isNonZero=true;
        for(int j=0; j<3; j++ ) {
            if (it != divPatternString.cend()) {
                int val=std::stoi(std::string(it,it+1));
                divPattern.push_back(val);
                isNonZero &= (val!=0);
            } else {
              std::cout << "Recursive division pattern " << divPatternString << " has wrong size\n";
              exit(-1);
            }
            it++;
        }
        if (!isNonZero){
          std::cout << "Recursive division pattern " << divPatternString << "contains 3 zeros in a step\n";
          exit(-1);
        }
    }

    // Check if the parameters make sense.
    int prodBFS = 1;

    for (int i = 0; i < r; i++) {
        int divM = divPattern[3*i];
        int divN = divPattern[3*i + 1];
        int divK = divPattern[3*i + 2];

        if (pattern[i] == 'b') {
          prodBFS *= divM * divN * divK;
        }
    }

    // Check if we are using too few processors!
    if (P < prodBFS) {
      std::cout << "Too few processors available for the given steps. The number of processors should be at least " << std::to_string(prodBFS) << ". Aborting the application.\n";
        exit(-1);
    }

    std::cout<<"Benchmarking "<<m<<"*"<<n<<"*"<<k<<" multiplication using "
        <<P<<" processes"<<std::endl;
    std::cout<<"Division pattern is: "<<pattern<<" - "
        <<divPatternString<<std::endl;

    multiply(m, n, k, P, r, pattern.cbegin(), divPattern.cbegin());

    return 0;
}
