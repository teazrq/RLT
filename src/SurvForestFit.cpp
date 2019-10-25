//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Utility.h"
# include "survForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List SurvForestUniFit(arma::mat& X,
          					 arma::uvec& Y,
          					 arma::uvec& Censor,
          					 arma::uvec& Ncat,
          					 List& param,
          					 List& RLTparam,
          					 arma::vec& obsweight,
          					 arma::vec& varweight,
          					 int usecores,
          					 int verbose)
{

  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF RLT Survival///" << std::endl;
  
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
  std::vector<Surv_Uni_Tree_Class> Forest(ntrees);

  arma::imat ObsTrack(N, ntrees, fill::zeros);

  arma::field<arma::field<arma::uvec>> NodeRegi(ntrees);
  
  vec VarImp(P, fill::zeros);

  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // prediction matrix
  cube Pred;
  
  // start to fit the model
  Surv_Uni_Forest_Build((const arma::mat&) X,
            					   (const arma::uvec&) Y,
            					   (const arma::uvec&) Censor,
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
  
  DEBUG_Rcout << "  --- prediction " << Pred.col(0) << std::endl;
  
  // save tree structure to arma::field
  
  //List Forest_R = surv_uni_convert_forest_to_r(Forest);

  // return subjects to R
  
  List ReturnList;

  //ReturnList["FittedForest"] = Forest_R;
  ReturnList["ObsTrack"] = ObsTrack;
  
  if (kernel_ready)
    ReturnList["NodeRegi"] = NodeRegi;
  else
    ReturnList["NodeRegi"] = R_NilValue;
  
  ReturnList["VarImp"] = VarImp;
  
  ReturnList["Prediction"] = mean(Pred, 1);
  
  umat inbag = (ObsTrack == 0);
  // ReturnList["OOBPrediction"] = sum(Pred % inbag, 1) / sum(inbag, 1);
  
  return ReturnList;
}




