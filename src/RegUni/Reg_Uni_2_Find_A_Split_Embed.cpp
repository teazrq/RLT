//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Find_A_Split_Embed(Split_Class& OneSplit,
                                const RLT_REG_DATA& REG_DATA,
                                const PARAM_GLOBAL& Param,
                                const uvec& obs_id,
                                uvec& var_id,
                                uvec& var_protect,
                                Rand& rngl)
{

  // set embedded model parameters 
  PARAM_GLOBAL Embed_Param;
  
  Embed_Param.N = obs_id.n_elem;
  Embed_Param.P = var_id.n_elem;
  Embed_Param.ntrees = Param.embed_ntrees;
  
  if (Param.embed_mtry > 1)
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
    
  // start fitting embedded model 
  
  // size_t N = Embed_Param.N;
  size_t P = Embed_Param.P;
  size_t ntrees = Embed_Param.ntrees;

  imat ObsTrack;
    
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
  
  // Initiate prediction objects
  vec Prediction;
  
  // VarImp
  vec VarImp(P, fill::zeros);
  
  // Run model fitting
  Reg_Uni_Forest_Build(REG_DATA,
                       REG_FOREST,
                       (const PARAM_GLOBAL&) Embed_Param,
                       obs_id,
                       (const uvec&) var_id,
                       ObsTrack,
                       true, // do prediction for VI
                       Prediction,
                       VarImp);
  
  var_id = var_id(sort_index(VarImp, "descend"));
  
  // protected variables
  size_t embed_protect = (Param.embed_protect < var_id.n_elem) ? Param.embed_protect : var_id.n_elem;
  var_protect = unique(join_cols(var_protect, var_id.subvec(0, embed_protect-1)));

  // calculate number of variables after muting
  size_t p_new;
  
  if (Param.embed_mute > 1)
    p_new = Embed_Param.P - Param.embed_mute;
  else
    p_new = Embed_Param.P - (size_t) Embed_Param.P * Param.embed_mute;
  
  if (p_new < 1) p_new = 1;

  // new variable list
  size_t var_best = var_id(0);
  var_id.resize(p_new);
  var_id = unique(join_cols(var_id, var_protect));

  // record and update 
  // the splitting rule 
  OneSplit.var = var_best;
  size_t split_gen = Param.split_gen;
  size_t split_rule = Param.split_rule;    
  size_t nsplit = Param.nsplit;
  double alpha = Param.alpha; 
  bool useobsweight = Param.useobsweight;  
  
  if (REG_DATA.Ncat(var_best) > 1) // categorical variable 
  {
    
    Reg_Uni_Split_Cat(OneSplit, 
                      obs_id, 
                      REG_DATA.X.unsafe_col(var_best), 
                      REG_DATA.Ncat(var_best),
                      REG_DATA.Y,
                      REG_DATA.obsweight,
                      0.0, // penalty
                      split_gen, 
                      split_rule, 
                      nsplit,
                      alpha, 
                      useobsweight,
                      rngl);
    
  }else{ // continuous variable
    
    Reg_Uni_Split_Cont(OneSplit,
                       obs_id,
                       REG_DATA.X.unsafe_col(var_best), 
                       REG_DATA.Y,
                       REG_DATA.obsweight,
                       0.0, // penalty
                       split_gen,
                       split_rule,
                       nsplit,
                       alpha,
                       useobsweight,
                       rngl);
    
  }
}

