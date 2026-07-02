//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"

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
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  size_t nsplit = Param.nsplit;
  size_t mtry = Param.mtry;
  size_t P = var_id.n_elem;
  size_t comb_size = std::min(P, Param.linear_comb);
  size_t N = obs_id.n_elem;
  
  // Tier selection: switch.size = 3 * embed_nmin
  size_t switch_size = 3 * Param.embed_nmin;
  
  if (N >= switch_size)
  {
    // ============================================================
    // Tier 1: Full Embed — fit internal forest, get VI, select top vars
    // ============================================================
    
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
      
    // Get the actual sample pool size (top linear_comb variables, regardless of threshold)
    size_t sample_pool = comb_size;  // This is min(P, linear_comb)
    
    // Update comb_valid to match sample_pool for later use
    size_t comb_valid = sample_pool;
    
    // New logic: randomly sample one variable from top sample_pool
    // Use UNIFORM sampling to give categorical variables a fair chance
    size_t sampled_idx = rngl.rand_sizet(0, sample_pool - 1);
    
    // Check if SAMPLED variable is categorical
    if (REG_DATA.Ncat(var_id(sampled_idx)) != 1)
    {
      // Sampled variable is categorical -> use single variable categorical split
      size_t j = var_id(sampled_idx);

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
      
      // calculate muting and protection (moved here for early return case)
      // Get threshold value (fraction of best VI for protection)
      double threshold = Param.embed_threshold;
      
      // Count how many of top embed_protect variables have VI >= threshold * best_vi
      size_t protect_valid = 0;
      for (size_t j = 0; j < std::min(Param.embed_protect, comb_valid); j++)
        protect_valid += (vi_embed(j) >= threshold * best_vi);
      
      // Protect only those passing threshold
      var_protect = unique(join_cols(var_protect, var_id.subvec(0, protect_valid - 1)));
      
      size_t p_new;
      if (Param.embed_mute >= 1)
        p_new = P - Param.embed_mute;
      else
        p_new = P * (1.0 - Param.embed_mute);
      
      p_new = std::max(p_new, comb_valid);
      
      var_id.resize(p_new);
      var_id = unique(join_cols(var_id, var_protect));
      
      return;
    }
    
    // Sampled variable is continuous -> collect ALL continuous variables in comb_valid
    // (skip categorical ones, don't stop at them)
    
    // Collect continuous variable indices and their VIs
    uvec cont_indices(comb_valid, fill::zeros);
    vec cont_vi(comb_valid, fill::zeros);
    size_t n_cont = 0;
    
    for (size_t j = 0; j < comb_valid; j++)
    {
      if (REG_DATA.Ncat(var_id(j)) == 1)
      {
        cont_indices(n_cont) = var_id(j);
        cont_vi(n_cont) = vi_embed(j);
        n_cont++;
      }
    }
    
    // Resize to actual count
    cont_indices.resize(n_cont);
    cont_vi.resize(n_cont);
    
    // At this point, top variable is continuous
    
    if (n_cont == 1) // single continuous split (only one continuous var in comb_valid)
    {
      size_t j = cont_indices(0);
      
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
    
    if (n_cont > 1) // linear combination split with multiple continuous variables
    {
      // for more than one variable, find best linear combination split
      Reg_Uni_Comb_Linear(OneSplit,
                          (const uvec&) cont_indices,
                          (const vec&) cont_vi,
                          REG_DATA,
                          Param,
                          obs_id,
                          rngl);
    }
    
    // calculate muting and protection
    // Get threshold value (fraction of best VI for protection)
    double threshold = Param.embed_threshold;
    
    // Count how many of top embed_protect variables have VI >= threshold * best_vi
    size_t protect_valid = 0;
    for (size_t j = 0; j < std::min(Param.embed_protect, comb_valid); j++)
        protect_valid += (vi_embed(j) >= threshold * best_vi);
    
    // Protect only those passing threshold
    var_protect = unique(join_cols(var_protect, var_id.subvec(0, protect_valid - 1)));
    
    // muting the low VIs
    size_t p_new;
    
    if (Param.embed_mute >= 1)
      p_new = P - Param.embed_mute;
    else
      p_new = P * (1.0 - Param.embed_mute);
    
    p_new = std::max(p_new, comb_valid);
    
    var_id.resize(p_new);
    var_id = unique(join_cols(var_id, var_protect));
    
  } // end Tier 1
  else
  {
    // ============================================================
    // Tier 2: Marginal Screen — univariate variance screening, no protection/muting
    // ============================================================
    
    // Sample mtry variables from var_id (or all if P <= mtry)
    size_t n_sample = std::min(P, mtry);
    uvec var_try = rngl.sample(var_id, n_sample);
    
    // Score each variable by its best univariate split (variance reduction)
    vec var_scores(n_sample, fill::zeros);
    
    for (size_t k = 0; k < n_sample; k++)
    {
      size_t j = var_try(k);
      
      if (REG_DATA.Ncat(j) > 1)
      {
        // Categorical variable — score it but mark separately
        var_scores(k) = -1.0; // negative score = categorical
      }
      else
      {
        // Continuous variable — find best split score
        Split_Class TempSplit;
        TempSplit.var = j;
        TempSplit.value = 0;
        TempSplit.score = -1;
        
        Reg_Uni_Split_Cont(TempSplit,
                           obs_id,
                           REG_DATA.X.unsafe_col(j),
                           REG_DATA.Y,
                           REG_DATA.obsweight,
                           0.0,
                           1, // variance reduction
                           nsplit,
                           alpha,
                           useobsweight,
                           rngl);
        
        var_scores(k) = TempSplit.score;
      }
    }
    
    // Collect continuous variables and sort by score
    uvec cont_indices(n_sample, fill::zeros);
    vec cont_scores(n_sample, fill::zeros);
    size_t n_cont = 0;
    
    for (size_t k = 0; k < n_sample; k++)
    {
      if (var_scores(k) > 0) // positive score = continuous with valid split
      {
        cont_indices(n_cont) = var_try(k);
        cont_scores(n_cont) = var_scores(k);
        n_cont++;
      }
    }
    
    cont_indices.resize(n_cont);
    cont_scores.resize(n_cont);
    
    if (n_cont == 0) return; // no valid splits
    
    // Sort continuous variables by score descending
    uvec score_rank = sort_index(cont_scores, "descend");
    cont_indices = cont_indices(score_rank);
    cont_scores = cont_scores(score_rank);
    
    // Take top comb_size continuous variables
    size_t n_use = std::min(n_cont, comb_size);
    
    // Dummy VI scores (all equal — marginal screen already ranked them)
    vec dummy_vi(n_use, fill::ones);
    
    if (n_use == 1) // single variable — just do univariate split
    {
      size_t j = cont_indices(0);
      
      Split_Class TempSplit;
      TempSplit.var = j;
      TempSplit.value = 0;
      TempSplit.score = -1;
      
      Reg_Uni_Split_Cont(TempSplit,
                         obs_id,
                         REG_DATA.X.unsafe_col(j),
                         REG_DATA.Y,
                         REG_DATA.obsweight,
                         0.0,
                         1,
                         nsplit,
                         alpha,
                         useobsweight,
                         rngl);
      
      OneSplit.var(0) = TempSplit.var;
      OneSplit.load(0) = 1;
      OneSplit.value = TempSplit.value;
      OneSplit.score = TempSplit.score;
    }
    else // multiple variables — send to LC method
    {
      uvec top_vars = cont_indices.subvec(0, n_use - 1);
      
      Reg_Uni_Comb_Linear(OneSplit,
                          (const uvec&) top_vars,
                          (const vec&) dummy_vi,
                          REG_DATA,
                          Param,
                          obs_id,
                          rngl);
    }
    
    // NO protection, NO muting for Tier 2
    // var_id passes to children unchanged
    
  } // end Tier 2
}
