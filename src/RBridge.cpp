//  **********************************
//  Reinforcement Learning Trees (RLT)
//  R bridging
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

//' @useDynLib RLT
//' @importFrom Rcpp sourceCpp
// [[Rcpp::export]]
arma::umat ARMA_EMPTY_UMAT()
{
  arma::umat temp;
  return temp;
}


//' @useDynLib RLT
//' @importFrom Rcpp sourceCpp
// [[Rcpp::export()]]
arma::vec ARMA_EMPTY_VEC()
{
  arma::vec temp;
  return temp;
}

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


//' @useDynLib RLT
//' @importFrom Rcpp sourceCpp
// [[Rcpp::export]]
arma::umat gen_ms_obs_track_mat_cpp(int ntrain, int sample_per_forest, int ntrees) {
  
  int ntrees_half = ntrees / 2;
  int k = sample_per_forest;
  arma::umat index_mat(ntrain, ntrees, arma::fill::zeros);
  
  for (int i = 0; i < ntrees_half; ++i) {
    arma::uvec rand_idx = arma::randperm(ntrain, 2 * k); // Random permutation of indices
    index_mat.submat(rand_idx.subvec(0, k - 1), arma::uvec(1, 1).fill(i)).fill(1);
    index_mat.submat(rand_idx.subvec(k, 2 * k - 1), arma::uvec(1, 1).fill(i + ntrees_half)).fill(1);
  }
  
  return index_mat;
}





