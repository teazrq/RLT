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
          					  arma::field<arma::field<arma::vec>>& NodeHaz,
          					  arma::field<arma::vec>& NodeSize,
          					  arma::mat& X,
          					  arma::uvec& Y,
          					  arma::uvec& Censor,
          					  arma::uvec& Ncat,
          					  List& param,
          					  arma::vec& obsweight,
          					  int NFail,
          					  bool kernel,
          					  int usecores,
          					  int verbose)
{
  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF RLT SURVIVAL PREDICTION///" << std::endl;

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
  std::vector<Surv_Uni_Tree_Class> Forest(ntrees);
  
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
      
      Forest[nt].NodeHaz.copy_size(NodeHaz[nt]);
      
      for (size_t j = 0; j < Forest[nt].NodeHaz.n_elem; j++)
        Forest[nt].NodeHaz[j] = vec(NodeHaz[nt][j].begin(), NodeHaz[nt][j].size(), false, true);
      
      Forest[nt].NodeSize = vec(NodeSize[nt].begin(), NodeSize[nt].size(), false, true);
    }
  }


  mat Pred = Surv_Uni_Forest_Pred(Forest,
                								 X,
                								 Ncat,
                								 NFail,
                								 kernel,
                								 usecores,
                								 verbose);

  
  List ReturnList;

  ReturnList["hazard"] = Pred;
  
  mat Surv(Pred);
  vec surv(N, fill::ones);
  vec Ones(N, fill::ones);
  
  for (size_t j=0; j < Surv.n_cols; j++)
  {
    surv = surv % (Ones - Surv.col(j)); //KM estimator
    Surv.col(j) = surv;
  }
  
  ReturnList["Survival"] = Surv;
  
  return ReturnList;
}
