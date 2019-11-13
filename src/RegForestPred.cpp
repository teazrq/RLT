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
          					  bool kernel,
          					  bool keep_all,          					  
          					  int usecores,
          					  int verbose)
{
  // check number of cores
  usecores = checkCores(usecores, verbose);

  // convert R object to forest
  
  Reg_Uni_Forest_Class REG_FOREST(NodeType, SplitVar, SplitValue, LeftNode, RightNode, NodeSize, NodeAve);
  
  mat PredAll;
  mat W;
  
  Reg_Uni_Forest_Pred(PredAll, 
                      W,
                      (const Reg_Uni_Forest_Class&) REG_FOREST,
          					  X,
          					  Ncat,
          					  kernel,
          					  usecores,
          					  verbose);
  
  List ReturnList;

  vec Pred;
  
  if (kernel)
  {
      Pred = sum(PredAll % W, 1) / sum(W, 1);
  }else{
      Pred = mean(PredAll, 1);
  }

  ReturnList["Prediction"] = Pred;
  
  if (keep_all)
    ReturnList["PredictionAll"] = PredAll;
      
  if (keep_all and kernel)
    ReturnList["WeightAll"] = W;
  
  return ReturnList;
}
