//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility//Utility.h"

using namespace Rcpp;
using namespace arma;

#ifndef ClaForest_Fun
#define ClaForest_Fun

// univariate tree split functions 

List ClaForestMultiFit(arma::mat& X,
                       arma::uvec& Y,
                       arma::uvec& Ncat,
                       List& param,
                       List& RLTparam,
                       arma::vec& obsweight,
                       arma::vec& varweight,
                       int usecores,
                       int verbose,
                       arma::umat& ObsTrack);

#endif
