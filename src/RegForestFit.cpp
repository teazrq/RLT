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

// other functions 
List reg_uni_convert_forest_to_r(std::vector<Reg_Uni_Tree_Class>& Forest);

// [[Rcpp::export()]]
List RegForestUniFit(arma::mat& X,
					 arma::vec& Y,
					 arma::uvec& Ncat,
					 List& param,
					 List& RLTparam,
					 arma::vec& obsweight,
					 arma::vec& varweight,
					 int usecores,
					 int verbose)
{

  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF RLT ///" << std::endl;
  
  // check number of cores
  usecores = checkCores(usecores, verbose);

  // readin parameters 
  PARAM_GLOBAL Param(param);
  PARAM_RLT Param_RLT;
  
  // create data objects
  size_t N = X.n_rows;
  size_t P = X.n_cols;
  size_t ntrees = Param.ntrees;
  bool kernel_ready = Param.kernel_ready;
  int seed = Param.seed;
    
  // initiate tree and other objects
  std::vector<Reg_Uni_Tree_Class> Forest(ntrees);

  arma::imat ObsTrack(N, ntrees, fill::zeros);

  arma::field<arma::field<arma::uvec>> NodeRegi(ntrees);
  
  vec VarImp(P, fill::zeros);

  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // prediction matrix
  mat Pred;
  
  // start to fit the model
  Reg_Uni_Forest_Build((const arma::mat&) X,
					   (const arma::vec&) Y,
					   (const arma::uvec&) Ncat,
					   (const PARAM_GLOBAL&) Param,
					   (const PARAM_RLT&) Param_RLT,
					   obsweight,
					   obs_id,
					   varweight,
					   var_id,
					   Forest,
					   ObsTrack,
					   Pred,
					   NodeRegi,
					   VarImp,
					   seed,
					   usecores,
					   verbose);

  DEBUG_Rcout << "  --- Finish fitting trees, start saving objects " << std::endl;
  
  // save tree structure to arma::field
  
  List Forest_R = reg_uni_convert_forest_to_r(Forest);

  // return subjects to R
  
  List ReturnList;

  ReturnList["FittedForest"] = Forest_R;
  ReturnList["ObsTrack"] = ObsTrack;
  
  if (kernel_ready)
    ReturnList["NodeRegi"] = NodeRegi;
  else
    ReturnList["NodeRegi"] = R_NilValue;
  
  ReturnList["VarImp"] = VarImp;
  
  ReturnList["Prediction"] = mean(Pred, 1);
  
  umat inbag = (ObsTrack == 0);
  ReturnList["OOBPrediction"] = sum(Pred % inbag, 1) / sum(inbag, 1);
  
  return ReturnList;
}


List reg_uni_convert_forest_to_r(std::vector<Reg_Uni_Tree_Class>& Forest)
{
  size_t ntrees = Forest.size();
  
  arma::field<arma::uvec> NodeType_Field(ntrees);
  arma::field<arma::uvec> SplitVar_Field(ntrees);
  arma::field<arma::vec> SplitValue_Field(ntrees);
  arma::field<arma::uvec> LeftNode_Field(ntrees);
  arma::field<arma::uvec> RightNode_Field(ntrees);
  arma::field<arma::vec> NodeAve_Field(ntrees);
  arma::field<arma::vec> NodeSize_Field(ntrees);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    NodeType_Field[nt] = uvec(Forest[nt].NodeType.begin(), Forest[nt].NodeType.size(), false, true);
    SplitVar_Field[nt] = uvec(Forest[nt].SplitVar.begin(), Forest[nt].SplitVar.size(), false, true);
    SplitValue_Field[nt] = vec(Forest[nt].SplitValue.begin(), Forest[nt].SplitValue.size(), false, true);
    LeftNode_Field[nt] = uvec(Forest[nt].LeftNode.begin(), Forest[nt].LeftNode.size(), false, true);
    RightNode_Field[nt] = uvec(Forest[nt].RightNode.begin(), Forest[nt].RightNode.size(), false, true);
    NodeAve_Field[nt] = vec(Forest[nt].NodeAve.begin(), Forest[nt].NodeAve.size(), false, true);
    NodeSize_Field[nt] = vec(Forest[nt].NodeSize.begin(), Forest[nt].NodeSize.size(), false, true);
  }
  
  return(List::create(Named("NodeType") = NodeType_Field,
                     Named("SplitVar") = SplitVar_Field,
                     Named("SplitValue") = SplitValue_Field,
                     Named("LeftNode") = LeftNode_Field,
                     Named("RightNode") = RightNode_Field,
                     Named("NodeAve") = NodeAve_Field,
                     Named("NodeSize") = NodeSize_Field));
}






