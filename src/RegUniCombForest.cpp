//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List RegUniCombForestFit(arma::mat& X,
            						 arma::vec& Y,
            						 arma::uvec& Ncat,
            						 arma::vec& obsweight,
            						 arma::vec& varweight,
            						 arma::imat& ObsTrack,
            						 List& param_r)
{
  // reading parameters 
  PARAM_GLOBAL Param;
  Param.PARAM_READ_R(param_r);
  
  if (Param.verbose) Param.print();
  
  // create data objects  
  RLT_REG_DATA REG_DATA(X, Y, Ncat, obsweight, varweight);
  
  size_t N = REG_DATA.X.n_rows;
  size_t P = REG_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  int obs_track = Param.obs_track;
  
  int importance = Param.importance;

  // initiate forest argument objects
  arma::field<arma::imat> SplitVar(ntrees);
  arma::field<arma::mat> SplitLoad(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::vec> NodeAve(ntrees);

  // Initiate forest object
  Reg_Uni_Comb_Forest_Class REG_FOREST(SplitVar,
                  										 SplitLoad,
                  										 SplitValue,
                  										 LeftNode,
                  										 RightNode,
                  										 NodeWeight,
                  										 NodeAve);
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // Initiate prediction objects
  vec Prediction;
  
  // VarImp
  vec VarImp;
  if (importance)
    VarImp.zeros(P);
  
  bool do_prediction = Param.replacement or (Param.resample_prob < 1);
  
  // Run model fitting
  Reg_Uni_Comb_Forest_Build((const RLT_REG_DATA&) REG_DATA,
              							REG_FOREST,
              							(const PARAM_GLOBAL&) Param,
              							obs_id,
              							var_id,
              							ObsTrack,
              							do_prediction,
              							Prediction,
              							VarImp);
  
  
  //initialize return objects
  List ReturnList;
  
  List Forest_R;
  
  //Save forest objects as part of return list  
  Forest_R["SplitVar"] = SplitVar;
  Forest_R["SplitLoad"] = SplitLoad;
  Forest_R["SplitValue"] = SplitValue;
  Forest_R["LeftNode"] = LeftNode;
  Forest_R["RightNode"] = RightNode;
  Forest_R["NodeWeight"] = NodeWeight;
  Forest_R["NodeAve"] = NodeAve;
  
  //Add to return list
  ReturnList["FittedForest"] = Forest_R;
  
  if (obs_track) ReturnList["ObsTrack"] = ObsTrack;
  if (importance) ReturnList["VarImp"] = VarImp;
  
  ReturnList["Prediction"] = Prediction;

  ReturnList["Error"] = mean(square(Prediction - Y));
  
  return ReturnList;
}

// [[Rcpp::export()]]
List RegUniCombForestPred(arma::field<arma::imat>& SplitVar,
                          arma::field<arma::mat>& SplitLoad,
                          arma::field<arma::vec>& SplitValue,
                          arma::field<arma::uvec>& LeftNode,
                          arma::field<arma::uvec>& RightNode,
                          arma::field<arma::vec>& NodeWeight,
                          arma::field<arma::vec>& NodeAve,
                          arma::mat& X,
                          arma::uvec& Ncat,
                          bool VarEst,
                          bool keep_all,
                          size_t usecores,
                          size_t verbose)
{
  // check number of cores
  usecores = checkCores(usecores, verbose);
  size_t N = X.n_rows;  
  
  // convert R object to forest
  
  Reg_Uni_Comb_Forest_Class REG_FOREST(SplitVar, 
                                       SplitLoad,
                                       SplitValue, 
                                       LeftNode, 
                                       RightNode, 
                                       NodeWeight, 
                                       NodeAve);

  size_t ntrees = REG_FOREST.SplitVarList.size();
  
  // Initialize prediction objects  
  mat PredAll(N, ntrees, fill::zeros);

#pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
      
      Reg_Uni_Comb_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                      REG_FOREST.SplitLoadList(nt),
                                      REG_FOREST.SplitValueList(nt),
                                      REG_FOREST.LeftNodeList(nt),
                                      REG_FOREST.RightNodeList(nt),
                                      REG_FOREST.NodeWeightList(nt),
                                      REG_FOREST.NodeAveList(nt));
      
      Find_Terminal_Node_Comb(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      PredAll.unsafe_col(nt).rows(real_id) = OneTree.NodeAve(TermNode);
    }
  }
  
  // Initialize return list
  List ReturnList;  
  
  ReturnList["Prediction"] = mean(PredAll, 1);  
  
  if (VarEst)
  {
    size_t B = (size_t) REG_FOREST.SplitVarList.size()/2;
    
    uvec firsthalf = linspace<uvec>(0, B-1, B);
    uvec secondhalf = linspace<uvec>(B, 2*B-1, B);
    
    // PredAll is n by ntrees
    vec SVar = var(PredAll, 0, 1); // norm_type = 0 means using n-1 as constant
    
    mat TreeDiff = PredAll.cols(firsthalf) - PredAll.cols(secondhalf);
    vec TreeVar = mean(square(TreeDiff), 1) / 2;
    
    vec Var = TreeVar*(1 + 1/2/B) - SVar*(1 - 1/2/B);

    ReturnList["Variance"] = Var;
  }
  
  // If keeping predictions for every tree  
  if (keep_all)
    ReturnList["PredictionAll"] = PredAll;  

  return ReturnList;
}











