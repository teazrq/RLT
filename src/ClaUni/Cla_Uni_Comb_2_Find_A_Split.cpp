//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Cla_Uni_Comb_Find_A_Split(Comb_Split_Class& OneSplit,
                               const RLT_CLA_DATA& Cla_DATA,
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
  size_t P = var_id.n_elem;
  size_t comb_size = std::min(P, Param.linear_comb);
  double threshold = Param.embed_threshold;
  size_t protect_size = std::min(Param.embed_protect, P);
  size_t nclass = Cla_DATA.nclass;
  
  //Embedded RF VI Screening Method
  vec vi_embed = Cla_Uni_Embed_Pre_Screen(Cla_DATA,
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
  
  // how many variables will pass the threshold
  // these variables will be protected
  size_t comb_valid = 0;
  for (size_t j = 0; j < comb_size; j++)
    comb_valid += (vi_embed(j) >= threshold * best_vi);
  
  // how many continuous variables in the linear combination
  size_t top_linear = 0;
  for (size_t j = 0; j < comb_valid; j++)
  {
    if (Cla_DATA.Ncat(var_id(j)) == 1)
      top_linear ++;
    else
      break;
  }

  // calculate splitting
  if (top_linear == 0) // categorical variable 
  {
    size_t j = var_id(0);
    
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    Cla_Uni_Split_Cat(TempSplit, 
                      obs_id, 
                      Cla_DATA.X.unsafe_col(j), 
                      Cla_DATA.Ncat(j),
                      Cla_DATA.Y,
                      Cla_DATA.obsweight,
                      nclass,
                      0.0, // penalty
                      split_gen, 
                      1, // gini index
                      nsplit,
                      alpha, 
                      useobsweight,
                      rngl);
    
    // record to linear combination split
    OneSplit.var(0) = TempSplit.var;
    OneSplit.load(0) = 1;
    OneSplit.value = TempSplit.value;
    OneSplit.score = TempSplit.score;
  }
  
  if (top_linear == 1) // single continuous split
  {
    size_t j = var_id(0);  
  
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;  

    Cla_Uni_Split_Cont(TempSplit,
                       obs_id,
                       Cla_DATA.X.unsafe_col(j), 
                       Cla_DATA.Y,
                       Cla_DATA.obsweight,
                       nclass,
                       0.0, // penalty
                       split_gen,
                       1,
                       nsplit,
                       alpha,
                       useobsweight,
                       rngl);
    
    // record to linear combination split
    OneSplit.var(0) = TempSplit.var;
    OneSplit.load(0) = 1;
    OneSplit.value = TempSplit.value;
    OneSplit.score = TempSplit.score;
  }

  if (top_linear > 1) // single continuous split
  {
    uvec split_var = var_id.subvec(0, top_linear-1);
    vec split_vi = vi_embed.subvec(0, top_linear-1);  
  
    
  
  }
  
  
  
  
  // how many variables will pass the threshold
  size_t protect_valid = 0;
  for (size_t j = 0; j < protect_size; j++)
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