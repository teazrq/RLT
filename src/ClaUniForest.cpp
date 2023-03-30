//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List ClaUniForestFit(arma::mat& X,
          					 arma::uvec& Y,
          					 arma::uvec& Ncat,
          					 size_t nclass,
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
  RLT_CLA_DATA CLA_DATA(X, Y, Ncat, nclass, obsweight, varweight);
  
  size_t N = CLA_DATA.X.n_rows;
  size_t P = CLA_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  int obs_track = Param.obs_track;

  int importance = Param.importance;

  // initiate forest argument objects
  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::mat> NodeProb(ntrees);
  
  //Initiate forest object
  Cla_Uni_Forest_Class CLA_FOREST(SplitVar, 
                                  SplitValue, 
                                  LeftNode, 
                                  RightNode,
                                  NodeWeight,
                                  NodeProb);
  
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // Initiate prediction objects
  mat Prediction;
  mat OOBPrediction;
  
  // VarImp
  vec VarImp;
  if (importance)
    VarImp.zeros(P);
  
  // Run model fitting
  Cla_Uni_Forest_Build((const RLT_CLA_DATA&) CLA_DATA,
                       CLA_FOREST,
                       (const PARAM_GLOBAL&) Param,
                       (const uvec&) obs_id,
                       (const uvec&) var_id,
                       ObsTrack,
                       true,
                       Prediction,
                       OOBPrediction,
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
  Forest_R["NodeProb"] = NodeProb;
  
  //Add to return list
  ReturnList["FittedForest"] = Forest_R;
  
  if (obs_track) ReturnList["ObsTrack"] = ObsTrack;
  if (importance) ReturnList["VarImp"] = VarImp;
  
  ReturnList["Prediction"] = index_max(Prediction, 1);
  ReturnList["OOBPrediction"] = index_max(OOBPrediction, 1);
  
  ReturnList["Prob"] = Prediction;
  ReturnList["OOBProb"] = OOBPrediction;
  
  return ReturnList;
}

// [[Rcpp::export()]]
List ClaUniForestPred(arma::field<arma::ivec>& SplitVar,
                      arma::field<arma::vec>& SplitValue,
                      arma::field<arma::uvec>& LeftNode,
                      arma::field<arma::uvec>& RightNode,
                      arma::field<arma::vec>& NodeWeight,
                      arma::field<arma::mat>& NodeProb,
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
  
  Cla_Uni_Forest_Class CLA_FOREST(SplitVar, 
                                  SplitValue, 
                                  LeftNode, 
                                  RightNode, 
                                  NodeWeight, 
                                  NodeProb);
  
  // Initialize prediction objects  
  cube PredAll;

  // run prediction
  Cla_Uni_Forest_Pred(PredAll,
                      (const Cla_Uni_Forest_Class&) CLA_FOREST,
                      X,
                      Ncat,
                      usecores,
                      verbose);
  
  // Initialize return list
  List ReturnList;
  
  uvec Pred(X.n_rows, fill::zeros);
  for (size_t i = 0; i < X.n_rows; i++)
    Pred(i) = index_max( mean(PredAll.slice(i), 0) );
  
  ReturnList["Prediction"] = Pred;
  
  // If keeping predictions for every tree  
  if (keep_all)
    ReturnList["PredictionAll"] = PredAll;
  
  return ReturnList;
}