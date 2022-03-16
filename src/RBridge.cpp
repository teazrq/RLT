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

//' @useDynLib RLT
//' @importFrom Rcpp sourceCpp
// [[Rcpp::export()]]
arma::uvec mysample(size_t Num, size_t min, size_t max, size_t seed)
{
  Rand rng(seed);
  
  uvec z = rng.sample(Num, min, max);
  
  return(z);
}
