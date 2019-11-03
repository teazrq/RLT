//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List RegForestUniPred(arma::field<arma::uvec>& NodeType,
          					  arma::field<arma::uvec>& SplitVar,
          					  arma::field<arma::vec>& SplitValue,
          					  arma::field<arma::uvec>& LeftNode,
          					  arma::field<arma::uvec>& RightNode,
          					  arma::field<arma::vec>& NodeAve,
          					  arma::field<arma::vec>& NodeSize,
          					  arma::mat& X,
          					  arma::vec& Y,
          					  arma::uvec& Ncat,
          					  List& param,
          					  arma::vec& obsweight,
          					  bool kernel,
          					  bool keep_all,          					  
          					  int usecores,
          					  int verbose)
{
  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF RLT REGRESSION PREDICTION///" << std::endl;

  // check number of cores
  usecores = checkCores(usecores, verbose);
  
  // readin parameters 
  PARAM_GLOBAL Param(param);
  size_t P = Param.P;
  
  if (P != X.n_cols)
    Rcpp::stop(" Dimension of testing data is different from training data ");
  
  size_t N = X.n_rows;
  size_t ntrees = NodeType.size();

  // convert R object to Reg_Uni_Tree_Class
  std::vector<Reg_Uni_Tree_Class> Forest(ntrees);
  
  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      Forest[nt].NodeType = uvec(NodeType[nt].begin(), NodeType[nt].size(), false, true);
      Forest[nt].SplitVar = uvec(SplitVar[nt].begin(), SplitVar[nt].size(), false, true);
      Forest[nt].SplitValue = vec(SplitValue[nt].begin(), SplitValue[nt].size(), false, true);
      Forest[nt].LeftNode = uvec(LeftNode[nt].begin(), LeftNode[nt].size(), false, true);
      Forest[nt].RightNode = uvec(RightNode[nt].begin(), RightNode[nt].size(), false, true);
      Forest[nt].NodeAve = vec(NodeAve[nt].begin(), NodeAve[nt].size(), false, true);
      Forest[nt].NodeSize = vec(NodeSize[nt].begin(), NodeSize[nt].size(), false, true);
    }
  }

  mat PredAll;
  mat W;
  
  Reg_Uni_Forest_Pred(PredAll, 
                      W, 
                      Forest,
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
  {
      ReturnList["PredictionAll"] = PredAll;
      
      if (kernel)
          ReturnList["WeightAll"] = W;
  }
  
  return ReturnList;
}
