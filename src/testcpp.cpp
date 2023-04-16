//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Random Forest Kernel
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
void testcpp(size_t n)
{
  // Generate all possible combinations
  std::vector<size_t> comb(n, 0);
  std::set<std::vector<size_t>> combinations;
  
  for (size_t i = 0; i < ((size_t) 1 << n); i++) {
    combinations.insert(comb);
    size_t j = n - 1;
    while (j >= 0 && comb[j] == 1) {
      comb[j] = 0;
      j--;
    }
    if (j >= 0) {
      comb[j] = 1;
    } else {
      break;
    }
  }
  
  // Output the visited vertices as binary vectors
  for (size_t i = 0; i < n; i++) {
    for (auto it = combinations.begin(); it != std::prev(combinations.end()); it++) {
      auto next = std::next(it);
      if ((*it)[i] != (*next)[i]) {
        for (size_t k = 0; k < n; k++) {
          RLTcout << (*it)[k] << " ";
        }
        RLTcout << std::endl;
      }
    }
  }
  
  for (size_t k = 0; k < n; k++) {
    RLTcout << combinations.rbegin()->at(k) << " ";
  }
  RLTcout << std::endl;

}