//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Stat Functions
//  **********************************

// my header file

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

#ifndef STAT_FUNCTIONS
#define STAT_FUNCTIONS

// sliced inverse regression
arma::mat sir(arma::mat& newX, 
              arma::vec& newY, 
              arma::vec& newW,
              bool useobsweight,
              size_t nslice);

// sliced average variance estimator 
arma::mat save(arma::mat& newX, 
               arma::vec& newY, 
               arma::vec& newW,
               bool useobsweight,
               size_t nslice);

// first eigenvector of pca
arma::mat first_pc(arma::mat& newX,
                   arma::vec& newW,
                   bool useobsweight);

// c-index 


double cindex_d(arma::vec& Y,
                arma::uvec& Censor,
                arma::vec& pred);

double cindex_i(arma::uvec& Y,
                arma::uvec& Censor,
                arma::vec& pred);

#endif