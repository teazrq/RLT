//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Find_A_Split_Embed(Split_Class& OneSplit,
                                 const RLT_SURV_DATA& SURV_DATA,
                                 const PARAM_GLOBAL& Param,
                                 const uvec& obs_id,
                                 uvec& var_id,
                                 uvec& var_protect,
                                 Rand& rngl)
{
  // parameters
  size_t P = var_id.n_elem;
  double threshold = Param.embed_threshold;
  
  // Embedded RF VI Screening Method
  vec vi_embed = Surv_Uni_Embed_Pre_Screen(SURV_DATA,
                                           Param,
                                           obs_id,
                                           var_id,
                                           rngl);
  
  // No comparable OOB outcomes means no evidence for selecting a variable.
  if (vi_embed.is_empty() || !vi_embed.is_finite())
    return;

  uvec vi_rank = stable_sort_index(vi_embed, "descend");
  var_id = var_id(vi_rank);
  vi_embed = vi_embed(vi_rank);
  
  // get best vi
  double best_vi = vi_embed(0);
  if (best_vi <= 0)  return;
  
  size_t best_j = var_id(0);

  // record the splitting variable
  OneSplit.var = best_j;

  // Use Surv_Uni_Find_A_Split with only the best variable
  // This reuses all collapse, precomputation, and split logic
  uvec single_var(1);
  single_var(0) = best_j;
  uvec obs_id_copy(obs_id);
  
  Surv_Uni_Find_A_Split(OneSplit,
                        SURV_DATA,
                        Param,
                        obs_id_copy,
                        single_var,
                        rngl);
  
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
  
  if (Param.embed_mute >= 1)
    p_new = P - Param.embed_mute;
  else
    p_new = P * (1.0 - Param.embed_mute);
  
  p_new = std::max(p_new, protect_valid);
  
  // new variable list
  var_id.resize(p_new);
  var_id = unique(join_cols(var_id, var_protect));

}
