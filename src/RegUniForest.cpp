//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List RegUniForestFit(arma::mat& X,
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
  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::vec> NodeAve(ntrees);
  
  //Initiate forest object
  Reg_Uni_Forest_Class REG_FOREST(SplitVar, 
                                  SplitValue, 
                                  LeftNode, 
                                  RightNode,
                                  NodeWeight,
                                  NodeAve);
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // Initiate oob-prediction objects
  vec Prediction;
  
  // VarImp
  vec VarImp;
  if (importance)
    VarImp.zeros(P);
  
  bool do_prediction = Param.replacement or (Param.resample_prob < 1);
    
  // Run model fitting
  Reg_Uni_Forest_Build((const RLT_REG_DATA&) REG_DATA,
                       REG_FOREST,
                       (const PARAM_GLOBAL&) Param,
                       (const uvec&) obs_id,
                       (const uvec&) var_id,
                       ObsTrack,
                       do_prediction,
                       Prediction,
                       VarImp);

  //initialize return objects
  List ReturnList;
  
  List Forest_R;

  //Save forest objects as part of return list  
  Forest_R["SplitVar"] = SplitVar;
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
List RegUniForestPred(arma::field<arma::ivec>& SplitVar,
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
  
  // convert R object to forest
  
  Reg_Uni_Forest_Class REG_FOREST(SplitVar, 
                                  SplitValue, 
                                  LeftNode, 
                                  RightNode, 
                                  NodeWeight, 
                                  NodeAve);
  
  // Initialize prediction objects  
  mat PredAll;

  // Run prediction
  Reg_Uni_Forest_Pred(PredAll,
                      (const Reg_Uni_Forest_Class&) REG_FOREST,
                      X,
                      Ncat,
                      usecores,
                      verbose);
  
  // Initialize return list
  List ReturnList;
  
  ReturnList["Prediction"] = mean(PredAll, 1);
  
  if (VarEst)
  {
    size_t B = (size_t) REG_FOREST.SplitVarList.size()/2;
    
    uvec firsthalf = linspace<uvec>(0, B-1, B);
    uvec secondhalf = linspace<uvec>(B, 2*B-1, B);
    
    vec SVar = var(PredAll, 0, 1); // norm_type = 1 means using n-1 as constant
    
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