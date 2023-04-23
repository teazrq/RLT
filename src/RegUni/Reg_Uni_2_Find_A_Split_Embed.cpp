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
  // parameters
  // size_t mtry = Param.mtry;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;
  size_t split_rule = Param.split_rule;
  size_t P = var_id.n_elem;
  double threshold = Param.embed_threshold;
  
  //Embedded RF VI Screening Method
  vec vi_embed = Reg_Uni_Embed_Pre_Screen(REG_DATA,
                                          Param,
                                          obs_id,
                                          var_id,
                                          rngl);
  
  uvec vi_rank = sort_index(vi_embed, "descend");
  var_id = var_id(vi_rank);
  vi_embed = vi_embed(vi_rank);
  
  // get best vi
  double best_vi = vi_embed(0);
  if (best_vi <= 0)  return;
  
  size_t best_j = var_id(0);

  // record and update 
  // the splitting rule 
  OneSplit.var = best_j;

  if (REG_DATA.Ncat(best_j) > 1) // categorical variable 
  {
    
    Reg_Uni_Split_Cat(OneSplit, 
                      obs_id, 
                      REG_DATA.X.unsafe_col(best_j), 
                      REG_DATA.Ncat(best_j),
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
                       REG_DATA.X.unsafe_col(best_j), 
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
  
  // calculate muting and protection
  size_t n_protect = std::min(Param.embed_protect, P);
  
  // how many variables will pass the threshold
  size_t protect_valid = 0;
  for (size_t j = 0; j < n_protect; j++)
    protect_valid += (vi_embed(j) >= threshold * best_vi);
  
  // for protecting the top variables
  var_protect = unique(join_cols(var_protect, var_id.subvec(0, protect_valid - 1)));
  
  // muting the low VIs
  size_t p_new;
  
  if (Param.embed_mute > 1)
    p_new = P - Param.embed_mute;
  else
    p_new = P * (1.0 - Param.embed_mute);
  
  p_new = std::max(p_new, protect_valid);
  
  // new variable list
  var_id.resize(p_new);
  var_id = unique(join_cols(var_id, var_protect));
  
}

