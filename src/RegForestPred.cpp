//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Utility.h"
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List RegForestUniPred(arma::field<arma::uvec>& NodeType,
          					  arma::field<arma::uvec>& SplitVar,
          					  arma::field<arma::vec>& SplitValue,
          					  arma::field<arma::uvec>& LeftNode,
          					  arma::field<arma::uvec>& RightNode,
          					  arma::field<arma::vec>& NodeSize,          					  
          					  arma::field<arma::vec>& NodeAve,
          					  arma::mat& X,
          					  arma::uvec& Ncat,
          					  bool keep_all,          					  
          					  int usecores,
          					  int verbose)
{
  // check number of cores
  usecores = checkCores(usecores, verbose);

  // convert R object to forest
  
  Reg_Uni_Forest_Class REG_FOREST(NodeType, SplitVar, SplitValue, LeftNode, RightNode, NodeSize, NodeAve);
  
  mat PredAll;
  
  Reg_Uni_Forest_Pred(PredAll,
                      (const Reg_Uni_Forest_Class&) REG_FOREST,
          					  X,
          					  Ncat,
          					  usecores,
          					  verbose);
  
  List ReturnList;

  ReturnList["Prediction"] = mean(PredAll, 1);
  
  if (keep_all)
    ReturnList["PredictionAll"] = PredAll;
  
  return ReturnList;
}
