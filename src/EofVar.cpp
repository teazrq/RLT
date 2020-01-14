//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Estimate the Expectation of Variance
//  **********************************

// my header file
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility/Utility.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List EofVar(arma::umat& ObsTrack,
            arma::mat& Pred,
            arma::uvec& C,
            int usecores,
            int verbose)
{
  DEBUG_Rcout << "-- calculate E(Var(Tree|C Shared)) ---" << std::endl;
  
  DEBUG_Rcout << C << std::endl;
  
  usecores = checkCores(usecores, verbose);
  
  size_t N = Pred.n_rows;
  size_t ntrees = Pred.n_cols;
  size_t length = C.n_elem;
  
  arma::mat Est(N, length, fill::zeros);
  arma::uvec allcounts(length, fill::zeros);
   
#pragma omp parallel num_threads(usecores)
{
  #pragma omp for schedule(dynamic)
  for (size_t l = 0; l < length; l++) // calculate all C values
  {
    size_t count = 0;
    
    for (size_t i = 0; i < (ntrees - 1); i++){
    for (size_t j = i+1; j < ntrees; j++){
      
      uvec pair = {i, j};
        
      if ( sum( min(ObsTrack.cols(pair), 1) ) == C(l) )
      {
        count++;
        
        Est.col(l) += 0.5 * square(Pred.col(i) - Pred.col(j));
      }
    }}
    
    Est.col(l) /= count;
    allcounts(l) = count;
  }
}

  DEBUG_Rcout << "-- total count  ---" << allcounts << std::endl;  
  DEBUG_Rcout << "-- all estimates  ---" << Est << std::endl; 

  List ReturnList;
  
  ReturnList["allcounts"] = allcounts;
  ReturnList["estimation"] = Est;
  
  return(ReturnList);
}








