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
                      int verbose,
                      arma::umat& ObsTrack)
{

  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF RLT SURVIVAL ///" << std::endl;
  
  // check number of cores
  usecores = checkCores(usecores, verbose);

  // readin parameters 
  PARAM_GLOBAL Param(param);
  if (verbose) Param.print();
  PARAM_RLT Param_RLT(RLTparam);
  if (verbose) Param_RLT.print();
  
  size_t NFail = max( Y(find(Censor == 1)) );  
  
  RLT_SURV_DATA SURV_DATA(X, Y, Censor, Ncat, NFail, obsweight, varweight);
  
  // create data objects
  size_t N = SURV_DATA.X.n_rows;
  size_t P = SURV_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  size_t seed = Param.seed;
  bool obs_track = Param.obs_track;
  int importance = Param.importance;
  
  // initiate forest
  arma::field<arma::uvec> NodeType(ntrees);
  arma::field<arma::uvec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeSize(ntrees);  
  arma::field<arma::field<arma::vec>> NodeHaz(ntrees);
  
  Surv_Uni_Forest_Class SURV_FOREST(NodeType, SplitVar, SplitValue, LeftNode, RightNode, NodeSize, NodeHaz);
  
  // other objects
  
  // VarImp
  vec VarImp;
  
  if (importance)
    VarImp.zeros(P);

  // prediction
  
  mat Prediction; // initialization means they will be calculated
  mat OOBPrediction;
  
  // mat Prediction(N, NFail, fill::zeros); // initialization means they will be calculated
  // mat OOBPrediction(N, NFail, fill::zeros);

  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // start to fit the model
  Surv_Uni_Forest_Build((const RLT_SURV_DATA&) SURV_DATA,
                        SURV_FOREST,
                        (const PARAM_GLOBAL&) Param,
                        (const PARAM_RLT&) Param_RLT,
                        obs_id,
                        var_id,
                        ObsTrack,
                        Prediction,
                        OOBPrediction,
                        VarImp,
                        seed,
                        usecores,
                        verbose);

  DEBUG_Rcout << "  --- Finish fitting trees, start saving objects " << std::endl;
  
  List ReturnList;
  
  List Forest_R;
  
  Forest_R["NodeType"] = NodeType;
  Forest_R["SplitVar"] = SplitVar;
  Forest_R["SplitValue"] = SplitValue;
  Forest_R["LeftNode"] = LeftNode;
  Forest_R["RightNode"] = RightNode;
  Forest_R["NodeSize"] = NodeSize;    
  Forest_R["NodeHaz"] = NodeHaz;

  
  ReturnList["FittedForest"] = Forest_R;
  
  if (obs_track) ReturnList["ObsTrack"] = ObsTrack;
  if (importance) ReturnList["VarImp"] = VarImp;
  
  ReturnList["Prediction"] = Prediction;
  ReturnList["OOBPrediction"] = OOBPrediction;
  
  // c index for model fitting 
  
  uvec nonNAs = find_finite(OOBPrediction.col(0));
  
  ReturnList["cindex"] = datum::nan;
  
  if (nonNAs.n_elem > 2)
  {
    vec oobpred(N, fill::zeros);
    
    for (auto i : nonNAs)
    {
      oobpred(i) = sum( cumsum( OOBPrediction.row(i) ) ); // sum of cumulative hazard as prediction
    }
    
    uvec oobY = Y(nonNAs);
    uvec oobC = Censor(nonNAs);
    vec oobP = oobpred(nonNAs);
    
    ReturnList["cindex"] =  cindex_i( oobY, oobC, oobP );
  }
  
  return ReturnList;
}