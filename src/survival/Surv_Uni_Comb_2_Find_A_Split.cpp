//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Comb_Split_A_Node
void Surv_Uni_Comb_Find_A_Split(Comb_Split_Class& OneSplit,
                                const RLT_SURV_DATA& SURV_DATA,
                                const PARAM_GLOBAL& Param,
                                uvec& obs_id,
                                uvec& var_id,
                                uvec& var_protect,
                                Rand& rngl)
{
  // parameters
  size_t mtry = Param.mtry;
  size_t P = var_id.n_elem;
  size_t N = obs_id.n_elem;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  bool usevarprob = Param.usevarprob;
  size_t nsplit = Param.nsplit;
  size_t split_rule = Param.split_rule;
  size_t comb_size = std::min(P, Param.linear_comb);

  // collapse Y into contiguous integers
  size_t NFail;
  uvec Y_collapse(N);
  uvec Censor_collapse(N);
  collapse(SURV_DATA.Y, SURV_DATA.Censor, Y_collapse, Censor_collapse, obs_id, NFail);

  if (NFail == 0)
    return;

  // Sample mtry variables from var_id
  size_t n_sample = std::min(P, mtry);
  uvec var_try;
  if (usevarprob)
  {
    vec varprob_try = SURV_DATA.varprob(var_id);
    var_try = rngl.weighted_sample(var_id, varprob_try, n_sample);
  }else{
    var_try = rngl.sample(var_id, n_sample);
  }

  // Pre-compute global fail/risk counts
  uvec All_Risk_Cum(NFail+1, fill::zeros);
  uvec All_Fail(NFail+1, fill::zeros);

  for (size_t i = 0; i < N; i++)
  {
    All_Risk_Cum(Y_collapse(i))++;
    All_Fail(Y_collapse(i)) += Censor_collapse(i);
  }

  cumsum_rev(All_Risk_Cum);

  // For suplogrank screening, we also need vec All_Risk and Temp_Vec
  vec All_Risk_vec(NFail+1, fill::zeros);
  vec Temp_Vec(NFail+1, fill::zeros);

  // Pre-compute weighted arrays if using observation weights
  vec node_wt(N, fill::ones);
  vec All_WFail(NFail+1, fill::zeros);
  vec All_WRisk_raw(NFail+1, fill::zeros);
  vec All_W2Risk_raw(NFail+1, fill::zeros);
  vec All_WRisk(NFail+1, fill::zeros);
  vec All_W2Risk(NFail+1, fill::zeros);

  if (useobsweight)
  {
    double wsum = 0;
    for (size_t i = 0; i < N; i++)
      wsum += SURV_DATA.obsweight(obs_id(i));
    double wbar = wsum / N;

    for (size_t i = 0; i < N; i++)
    {
      double wi = survival_node_weight(SURV_DATA.obsweight(obs_id(i)), wsum, wbar, N);
      node_wt(i) = wi;
      double w2i = wi * wi;

      All_WRisk_raw(Y_collapse(i)) += wi;
      All_WFail(Y_collapse(i)) += wi * Censor_collapse(i);
      All_W2Risk_raw(Y_collapse(i)) += w2i;
    }

    // Cumulative from right
    All_WRisk(NFail) = All_WRisk_raw(NFail);
    All_W2Risk(NFail) = All_W2Risk_raw(NFail);
    for (size_t j = NFail; j > 0; j--)
    {
      All_WRisk(j-1) = All_WRisk_raw(j-1) + All_WRisk(j);
      All_W2Risk(j-1) = All_W2Risk_raw(j-1) + All_W2Risk(j);
    }
  }

  // For suplogrank, convert All_Risk to vec
  if (split_rule == 2)
  {
    for (size_t j = 0; j <= NFail; j++)
      All_Risk_vec(j) = All_Risk_Cum(j);
    if (!useobsweight)
    {
      vec fail = conv_to<vec>::from(All_Fail);
      Temp_Vec = (1.0 - (fail - 1.0) / (All_Risk_vec - 1.0)) % fail / All_Risk_vec;
      Temp_Vec.elem(find(All_Risk_vec < 2)).zeros();
    }
  }

  // Score each variable by its best univariate split
  vec var_scores(n_sample, fill::zeros);

  for (size_t k = 0; k < n_sample; k++)
  {
    size_t j = var_try(k);

    if (SURV_DATA.Ncat(j) > 1)
    {
      // Categorical — mark with negative score
      var_scores(k) = -1.0;
    }else{

      Split_Class TempSplit;
      TempSplit.var = j;
      TempSplit.value = 0;
      TempSplit.score = -1;

      if (split_rule == 1) // logrank screening
      {
        if (useobsweight)
        {
          Surv_Uni_Logrank_Cont(TempSplit,
                                obs_id,
                                SURV_DATA.X.unsafe_col(j),
                                Y_collapse,
                                Censor_collapse,
                                NFail,
                                All_Fail,
                                All_Risk_Cum,
                                nsplit,
                                alpha,
                                rngl,
                                true,
                                node_wt,
                                All_WFail,
                                All_WRisk,
                                All_W2Risk);
        }else{
          Surv_Uni_Logrank_Cont(TempSplit,
                                obs_id,
                                SURV_DATA.X.unsafe_col(j),
                                Y_collapse,
                                Censor_collapse,
                                NFail,
                                All_Fail,
                                All_Risk_Cum,
                                nsplit,
                                alpha,
                                rngl);
        }
      }
      else if (split_rule == 2) // suplogrank screening
      {
        if (useobsweight)
        {
          Surv_Uni_SupLogrank_Cont(TempSplit,
                                   obs_id,
                                   SURV_DATA.X.unsafe_col(j),
                                   Y_collapse,
                                   Censor_collapse,
                                   NFail,
                                   All_Fail,
                                   All_Risk_vec,
                                   nsplit,
                                   alpha,
                                   rngl,
                                   node_wt,
                                   All_WFail,
                                   All_WRisk,
                                   All_W2Risk);
        }else{
          Surv_Uni_Split_Cont(TempSplit,
                              obs_id,
                              SURV_DATA.X.unsafe_col(j),
                              Y_collapse,
                              Censor_collapse,
                              NFail,
                              All_Fail,
                              All_Risk_vec,
                              Temp_Vec,
                              SURV_DATA.obsweight,
                              split_rule,
                              nsplit,
                              alpha,
                              useobsweight,
                              rngl);
        }
      }
      else // split_rule == 3 (coxgrad): use logrank screening
      {
        if (useobsweight)
        {
          Surv_Uni_Logrank_Cont(TempSplit,
                                obs_id,
                                SURV_DATA.X.unsafe_col(j),
                                Y_collapse,
                                Censor_collapse,
                                NFail,
                                All_Fail,
                                All_Risk_Cum,
                                nsplit,
                                alpha,
                                rngl,
                                true,
                                node_wt,
                                All_WFail,
                                All_WRisk,
                                All_W2Risk);
        }else{
          Surv_Uni_Logrank_Cont(TempSplit,
                                obs_id,
                                SURV_DATA.X.unsafe_col(j),
                                Y_collapse,
                                Censor_collapse,
                                NFail,
                                All_Fail,
                                All_Risk_Cum,
                                nsplit,
                                alpha,
                                rngl);
        }
      }

      var_scores(k) = TempSplit.score;
    }
  }

  // Collect continuous variables and sort by score
  uvec cont_indices(n_sample, fill::zeros);
  vec cont_scores(n_sample, fill::zeros);
  size_t n_cont = 0;

  for (size_t k = 0; k < n_sample; k++)
  {
    if (var_scores(k) > 0)
    {
      cont_indices(n_cont) = var_try(k);
      cont_scores(n_cont) = var_scores(k);
      n_cont++;
    }
  }

  cont_indices.resize(n_cont);
  cont_scores.resize(n_cont);

  if (n_cont == 0) return;

  // Sort continuous variables by score descending
  uvec score_rank = sort_index(cont_scores, "descend");
  cont_indices = cont_indices(score_rank);
  cont_scores = cont_scores(score_rank);

  // Take top comb_size continuous variables
  size_t n_use = std::min(n_cont, comb_size);

  // VI scores (use screening scores)
  vec vi_scores = cont_scores.subvec(0, n_use - 1);

  if (n_use == 1) // single variable — just do univariate split
  {
    // Copy screening result directly
    OneSplit.var(0) = cont_indices(0);
    OneSplit.load(0) = 1;
    OneSplit.value = 0;
    OneSplit.score = cont_scores(0);

    // Re-run to get the actual split point
    size_t j = cont_indices(0);
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;

    if (split_rule == 1)
    {
      if (useobsweight)
      {
        Surv_Uni_Logrank_Cont(TempSplit,
                              obs_id,
                              SURV_DATA.X.unsafe_col(j),
                              Y_collapse,
                              Censor_collapse,
                              NFail,
                              All_Fail,
                              All_Risk_Cum,
                              nsplit,
                              alpha,
                              rngl,
                              true,
                              node_wt,
                              All_WFail,
                              All_WRisk,
                              All_W2Risk);
      }else{
        Surv_Uni_Logrank_Cont(TempSplit,
                              obs_id,
                              SURV_DATA.X.unsafe_col(j),
                              Y_collapse,
                              Censor_collapse,
                              NFail,
                              All_Fail,
                              All_Risk_Cum,
                              nsplit,
                              alpha,
                              rngl);
      }
    }
    else if (split_rule == 2)
    {
      if (useobsweight)
      {
        Surv_Uni_SupLogrank_Cont(TempSplit,
                                 obs_id,
                                 SURV_DATA.X.unsafe_col(j),
                                 Y_collapse,
                                 Censor_collapse,
                                 NFail,
                                 All_Fail,
                                 All_Risk_vec,
                                 nsplit,
                                 alpha,
                                 rngl,
                                 node_wt,
                                 All_WFail,
                                 All_WRisk,
                                 All_W2Risk);
      }else{
        Surv_Uni_Split_Cont(TempSplit,
                            obs_id,
                            SURV_DATA.X.unsafe_col(j),
                            Y_collapse,
                            Censor_collapse,
                            NFail,
                            All_Fail,
                            All_Risk_vec,
                            Temp_Vec,
                            SURV_DATA.obsweight,
                            split_rule,
                            nsplit,
                            alpha,
                            useobsweight,
                            rngl);
      }
    }
    else // split_rule == 3
    {
      if (useobsweight)
      {
        Surv_Uni_Logrank_Cont(TempSplit,
                              obs_id,
                              SURV_DATA.X.unsafe_col(j),
                              Y_collapse,
                              Censor_collapse,
                              NFail,
                              All_Fail,
                              All_Risk_Cum,
                              nsplit,
                              alpha,
                              rngl,
                              true,
                              node_wt,
                              All_WFail,
                              All_WRisk,
                              All_W2Risk);
      }else{
        Surv_Uni_Logrank_Cont(TempSplit,
                              obs_id,
                              SURV_DATA.X.unsafe_col(j),
                              Y_collapse,
                              Censor_collapse,
                              NFail,
                              All_Fail,
                              All_Risk_Cum,
                              nsplit,
                              alpha,
                              rngl);
      }
    }

    OneSplit.var(0) = TempSplit.var;
    OneSplit.load(0) = 1;
    OneSplit.value = TempSplit.value;
    OneSplit.score = TempSplit.score;
  }
  else // multiple variables — send to LC method
  {
    uvec top_vars = cont_indices.subvec(0, n_use - 1);

    Surv_Uni_Comb_Linear(OneSplit,
                         (const uvec&) top_vars,
                         (const vec&) vi_scores,
                         SURV_DATA,
                         Param,
                         obs_id,
                         rngl);
  }
}
