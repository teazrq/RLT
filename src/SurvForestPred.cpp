//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "survForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List SurvForestUniPred(arma::field<arma::uvec>& NodeType,
          					   arma::field<arma::uvec>& SplitVar,
          					   arma::field<arma::vec>& SplitValue,
          					   arma::field<arma::uvec>& LeftNode,
          					   arma::field<arma::uvec>& RightNode,
          					   arma::field<arma::vec>& NodeSize,
          					   arma::field<arma::field<arma::vec>>& NodeHaz,
          					   arma::mat& X,
          					   arma::uvec& Ncat,
          					   size_t NFail,
          					   arma::uvec& treeindex,
          					   bool keep_all,
          					   int usecores,
          					   int verbose)
{
  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF RLT SURVIVAL PREDICTION///" << std::endl;

  // check number of cores
  usecores = checkCores(usecores, verbose);
  
  Surv_Uni_Forest_Class SURV_FOREST(NodeType, SplitVar, SplitValue, LeftNode, RightNode, NodeSize, NodeHaz);

  cube Pred;
  
  // predict 
  
  Surv_Uni_Forest_Pred(Pred,
                       SURV_FOREST,
      								 X,
      								 Ncat,
      								 NFail,
      								 treeindex,
      								 usecores,
      								 verbose);

  // get hazard function by averaging all trees (not weighted, need to update)
  // Pred is NFail by ntrees by N
  
  // Rcout << "-- first subject is " << X.row(0) << std::endl;
  
  mat H(Pred.n_slices, Pred.n_rows);
  
#pragma omp parallel num_threads(usecores)
#pragma omp for schedule(static)
  for (size_t i = 0; i < Pred.n_slices; i++)
  {
    H.row(i) = mean(Pred.slice(i), 1).t();
  }
  
  List ReturnList;

  ReturnList["hazard"] = H;  
  
  mat Surv(H);
  vec surv(H.n_rows, fill::ones);
  vec Ones(H.n_rows, fill::ones);
  
  for (size_t j=0; j < Surv.n_cols; j++)
  {
    surv = surv % (Ones - Surv.col(j)); //KM estimator
    Surv.col(j) = surv;
  }
  
  ReturnList["Survival"] = Surv;  
  
  if (keep_all)
    ReturnList["Allhazard"] = Pred;
  
  return ReturnList;
}
