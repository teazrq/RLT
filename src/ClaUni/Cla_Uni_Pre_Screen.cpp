//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Embedded RF VI Screening Method
vec Cla_Uni_Embed_Pre_Screen(const RLT_CLA_DATA& Cla_DATA,
                             const PARAM_GLOBAL& Param,
                             const uvec& obs_id,
                             const uvec& var_id,
                             Rand& rngl)
{
  // set embedded model parameters 
  PARAM_GLOBAL Embed_Param;
  
  Embed_Param.N = obs_id.n_elem;
  Embed_Param.P = var_id.n_elem;
  Embed_Param.ntrees = Param.embed_ntrees;
  
  // mtry can be < 1 or >= 1
  if (Param.embed_mtry >= 1)
    Embed_Param.mtry = (size_t) Param.embed_mtry;
  else
    Embed_Param.mtry = (size_t) Embed_Param.P * Param.embed_mtry;
  
  Embed_Param.nmin = Param.embed_nmin;
  Embed_Param.split_gen = Param.embed_split_gen;
  Embed_Param.nsplit = Param.embed_nsplit;
  Embed_Param.replacement = Param.embed_replacement;
  Embed_Param.resample_prob = Param.embed_resample_prob;
  Embed_Param.useobsweight = Param.useobsweight;
  Embed_Param.usevarweight = Param.usevarweight;
  Embed_Param.importance = 2; // use distributed variable importance for stability
  
  Embed_Param.ncores = 1;
  Embed_Param.verbose = 0;
  Embed_Param.seed = rngl.rand_sizet(0, INT_MAX);  
  
  // Embed_Param.split_rule = 1 always use default? 
  
  // size_t N = Embed_Param.N;
  size_t P = Embed_Param.P;
  size_t ntrees = Embed_Param.ntrees;  
  
  imat ObsTrack;
  
  // initiate uni forest objects
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
  
  // Initiate prediction objects
  mat Prediction;
  
  // VarImp
  vec VarImp(P, fill::zeros);

  // Run model fitting
  Cla_Uni_Forest_Build(Cla_DATA,
                       CLA_FOREST,
                       (const PARAM_GLOBAL&) Embed_Param,
                       obs_id,
                       var_id,
                       ObsTrack,
                       true, // do prediction for VI
                       Prediction,
                       VarImp);

  return VarImp;
}
