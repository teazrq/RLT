//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Reg_Uni_Comb_Find_A_Split(Comb_Split_Class& OneSplit,
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
  size_t P = var_id.n_elem;
  size_t comb_size = std::min(P, Param.linear_comb);
  double comb_threshold = Param.embed_threshold;
  
  //Embedded RF VI Screening Method
  vec vi_embed = Reg_Uni_Embed_Pre_Screen(REG_DATA,
                                          Param,
                                          obs_id,
                                          var_id,
                                          rngl);

  // sort vi
  uvec vi_rank = sort_index(vi_embed, "descend");
  var_id = var_id(vi_rank);
  vi_embed = vi_embed(vi_rank);
  
  // get best vi
  double best_vi = vi_embed(0);
  if (best_vi <= 0)  return;
    
  // how many variables will pass the threshold
  size_t comb_valid = 0;
  for (size_t j = 0; j < comb_size; j++)
    comb_valid += (vi_embed(j) >= comb_threshold * best_vi);

  // how many continuous variables in the linear combination
  size_t top_linear = 0;
  for (size_t j = 0; j < comb_valid; j++)
  {
    if (REG_DATA.Ncat(var_id(j)) == 1)
      top_linear ++;
    else
      break;
  }
  
  // calculate splitting
  if (top_linear == 0) // categorical split
  {
    size_t j = var_id(0);

    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    Reg_Uni_Split_Cat(TempSplit, 
                      obs_id, 
                      REG_DATA.X.unsafe_col(j), 
                      REG_DATA.Ncat(j),
                      REG_DATA.Y, 
                      REG_DATA.obsweight, 
                      0.0, // penalty
                      split_gen,
                      1, // univariate splitting rule (var)
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
    
    Reg_Uni_Split_Cont(TempSplit,
                       obs_id,
                       REG_DATA.X.unsafe_col(j), 
                       REG_DATA.Y,
                       REG_DATA.obsweight,
                       0.0, // penalty
                       split_gen,
                       1, // univariate splitting rule (var)
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
    
    // for more than one variable, find best linear combination split
    Reg_Uni_Comb_Linear(OneSplit,
                        (const uvec&) split_var,
                        REG_DATA,
                        Param,
                        obs_id,
                        rngl);
  }
  
  // calculate muting and protection

  // for protecting the top variables
  size_t n_protect = std::min(Param.embed_protect, comb_valid);
  var_protect = unique(join_cols(var_protect, var_id.subvec(0, n_protect - 1)));
  
  // muting the low VIs
  size_t p_new;
  
  if (Param.embed_mute > 1)
    p_new = P - Param.embed_mute;
  else
    p_new = P * (1.0 - Param.embed_mute);
  
  p_new = std::max(p_new, comb_valid);
  
  var_id.resize(p_new);
  var_id = unique(join_cols(var_id, var_protect));
}