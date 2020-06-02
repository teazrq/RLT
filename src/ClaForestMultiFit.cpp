//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Utility.h"
# include "claForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List ClaForestMultiFit(arma::mat& X,
          					   arma::uvec& Y,
          					   arma::uvec& Ncat,
          					   List& param,
          					   List& RLTparam,
          					   arma::vec& obsweight,
          					   arma::vec& varweight,
          					   int usecores,
          					   int verbose,
          					   arma::umat& ObsTrack)
{
  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF Classification Multi ///" << std::endl;

  return 0;
}