//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// Cox PH Loading: fit Cox PH model on node data to get direction v = beta
// With optional observation weights
void Surv_CoxPH_Loading(mat& V,
                         const mat& newX,
                         const uvec& newY,
                         const uvec& Censor,
                         const vec& split_vi,
                         size_t NFail,
                         size_t P,
                         size_t N,
                         const vec& obs_weight,
                         bool useobsweight)
{
  // Fit weighted Cox PH model h(t|X) = h0(t) exp(X beta)
  // Use Newton-Raphson on the weighted partial likelihood to estimate beta
  // Weighted: U = sum_i delta_i * omega_i * [X_i - Xbar_omega(t_i)]
  // where Xbar_omega(t) = sum_{j in R(t)} omega_j * X_j * exp(X_j'beta) / sum_{j in R(t)} omega_j * exp(X_j'beta)
  // Return V as P x 1 matrix (single direction)
  
  V.zeros(P, 1);
  
  if (N < 5 || NFail == 0 || P == 0)
  {
    V(0, 0) = 1.0;
    return;
  }
  
  // Get weights (default to 1.0 if not using weights)
  vec wi(N, fill::ones);
  if (useobsweight)
  {
    for (size_t i = 0; i < N; i++)
      wi(i) = obs_weight(i);
  }
  
  // Compute at-risk counts and failure counts for the node
  uvec RiskCount(NFail + 1, fill::zeros);
  uvec FailCount(NFail + 1, fill::zeros);
  
  for (size_t i = 0; i < N; i++)
  {
    RiskCount(newY(i))++;
    FailCount(newY(i)) += Censor(i);
  }
  
  // Cumulative at-risk counts (from right)
  vec RiskCum(NFail + 1, fill::zeros);
  for (size_t j = 0; j < NFail + 1; j++)
    RiskCum(j) = RiskCount(j);
  
  for (size_t j = NFail; j > 0; j--)
    RiskCum(j-1) += RiskCum(j);
  
  // Newton-Raphson for weighted Cox PH partial likelihood
  // Score: U = sum_i delta_i * omega_i * [X_i - Xbar_omega(t_i)]
  // Information: I = sum_i delta_i * omega_i * [sum_{j in R(t_i)} omega_j*X_j*X_j'*w_j / W - (Xbar_omega)(Xbar_omega)']
  // where w_j = exp(X_j'beta), W = sum omega_j * w_j
  
  vec beta(P, fill::zeros); // start at zero
  
  size_t max_iter = 20;
  double tol = 1e-6;
  
  bool converged = false;
  
  for (size_t iter = 0; iter < max_iter; iter++)
  {
    // compute exp(X beta) for all observations
    vec exp_xb = exp(newX * beta);
    
    // clamp to prevent overflow
    exp_xb = clamp(exp_xb, 1e-10, 1e10);
    
    vec score(P, fill::zeros);
    mat info(P, P, fill::zeros);
    
    for (size_t t = 1; t <= NFail; t++)
    {
      if (FailCount(t) == 0) continue;
      
      // at-risk set at time t — compute weighted sums
      double W = 0; // sum of omega_j * exp(X beta) for at-risk
      vec WX(P, fill::zeros); // sum of omega_j * X_j * exp(X_j beta) for at-risk
      mat WXX(P, P, fill::zeros); // sum of omega_j * X_j X_j' * exp(X_j beta) for at-risk
      
      for (size_t i = 0; i < N; i++)
      {
        if (newY(i) >= t) // at risk at time t
        {
          double w = wi(i) * exp_xb(i);
          W += w;
          WX += w * newX.row(i).t();
          WXX += w * (newX.row(i).t() * newX.row(i));
        }
      }
      
      if (W < 1e-10) continue;
      
      vec xbar = WX / W;
      mat xxt = WXX / W;
      
      // For each failure at time t, add weighted contribution
      for (size_t i = 0; i < N; i++)
      {
        if (newY(i) == t && Censor(i) == 1)
        {
          score += wi(i) * (newX.row(i).t() - xbar);
          info += wi(i) * (xxt - xbar * xbar.t());
        }
      }
    }
    
    // Ridge regularization for numerical stability
    info.diag() += 1e-6 * trace(info) / P;
    
    // Newton update: beta += I^{-1} * score
    vec delta;
    bool solve_ok = true;
    try {
      delta = solve(info, score, solve_opts::fast);
    } catch (...) {
      solve_ok = false;
    }
    
    if (!solve_ok || delta.has_nan() || delta.has_inf())
      break;
    
    beta += delta;
    
    // check convergence
    if (norm(delta) < tol * (norm(beta) + tol))
    {
      converged = true;
      break;
    }
  }
  
  // Check result
  if (!converged || beta.has_nan() || beta.has_inf())
  {
    // Fallback: use weighted score statistic direction (one-step)
    // Score at beta=0: U = sum_i delta_i * omega_i * [X_i - Xbar_omega(t_i)]
    vec score(P, fill::zeros);
    
    for (size_t t = 1; t <= NFail; t++)
    {
      if (FailCount(t) == 0) continue;
      
      double wt_at_risk = 0; // sum of omega_j for at-risk
      vec wt_x(P, fill::zeros); // sum of omega_j * X_j for at-risk
      
      for (size_t i = 0; i < N; i++)
      {
        if (newY(i) >= t)
        {
          wt_at_risk += wi(i);
          wt_x += wi(i) * newX.row(i).t();
        }
      }
      
      if (wt_at_risk < 1e-10) continue;
      
      vec mean_x = wt_x / wt_at_risk;
      
      for (size_t i = 0; i < N; i++)
      {
        if (newY(i) == t && Censor(i) == 1)
          score += wi(i) * (newX.row(i).t() - mean_x);
      }
    }
    
    if (score.has_nan() || score.has_inf() || norm(score) < 1e-10)
    {
      V(0, 0) = 1.0;
      return;
    }
    
    beta = score;
  }
  
  // Apply VI weighting
  V.col(0) = beta % sqrt(split_vi);
  
  // Sign convention: max absolute element should be positive
  uword max_idx = beta.index_max();
  if (V(max_idx, 0) < 0)
    V.col(0) = -V.col(0);
  
  // Normalize
  double vnorm = norm(V.col(0));
  if (vnorm > 1e-10)
    V.col(0) /= vnorm;
  else
  {
    V.zeros(P, 1);
    V(0, 0) = 1.0;
  }
}

// Main linear combination split function for survival
void Surv_Uni_Comb_Linear(Comb_Split_Class& OneSplit,
                           const uvec& split_var,
                           const vec& split_vi,
                           const RLT_SURV_DATA& SURV_DATA,
                           const PARAM_GLOBAL& Param,
                           const uvec& obs_id,
                           Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t P = split_var.n_elem;
  bool useobsweight = Param.useobsweight;
  
  if (N < 2 || P == 0)
  {
    OneSplit.var(0) = split_var.n_elem > 0 ? split_var(0) : 0;
    OneSplit.load(0) = 1.0;
    OneSplit.value = 0;
    OneSplit.score = -1;
    return;
  }
  
  // Extract node data
  mat newX(SURV_DATA.X(obs_id, split_var));
  uvec newY = SURV_DATA.Y(obs_id);
  uvec newCensor = SURV_DATA.Censor(obs_id);
  
  // Observation weights for this node (normalized by mean)
  vec node_wt(N, fill::ones);
  if (useobsweight)
  {
    double wsum = 0;
    for (size_t i = 0; i < N; i++)
      wsum += SURV_DATA.obsweight(obs_id(i));
    double wbar = wsum / N;
    for (size_t i = 0; i < N; i++)
      node_wt(i) = survival_node_weight(SURV_DATA.obsweight(obs_id(i)), wsum, wbar, N);
  }
  
  // Manually collapse Y into contiguous integers for this node
  size_t NFail;
  uvec Y_collapse(N);
  uvec Censor_collapse(N);
  
  // Sort by Y then by Censor for collapse
    uvec sort_idx = sort_index(newY);
    uvec sorted_newY = newY(sort_idx);
    uvec sorted_newCensor = newCensor(sort_idx);
    
    size_t timepoint = 0;
    size_t current_y = sorted_newY(0);
    size_t i = 0;
    
    if (sorted_newCensor(0) == 1)
    {
      timepoint = 1;
      Y_collapse(sort_idx(0)) = 1;
      Censor_collapse(sort_idx(0)) = 1;
      i = 1;
    }else{
      while(i < N && sorted_newCensor(i) == 0)
      {
        Y_collapse(sort_idx(i)) = timepoint;
        Censor_collapse(sort_idx(i)) = 0;
        current_y = sorted_newY(i);
        i++;
      }
    }
    
    for(; i < N; i++)
    {
      if (sorted_newY(i) > current_y && sorted_newCensor(i) == 1)
      {
        timepoint++;
        current_y = sorted_newY(i);
      }
      Y_collapse(sort_idx(i)) = timepoint;
      Censor_collapse(sort_idx(i)) = sorted_newCensor(i);
    }
    
    NFail = timepoint;
  
  if (NFail == 0)
  {
    OneSplit.var(0) = split_var(0);
    OneSplit.load(0) = 1.0;
    OneSplit.score = -1;
    return;
  }
  
  // Compute loading direction
  // linear_comb_method: 1 = CoxPH (default), 2 = naive (score statistic)
  mat V;
  size_t lc_method = Param.linear_comb_method;
  
  if (lc_method == 2 || N < 10) // naive or small sample fallback
  {
    // Weighted score statistic direction
    V.zeros(P, 1);
    vec score_dir(P, fill::zeros);
    
    // Compute at-risk counts
    uvec RiskCount(NFail + 1, fill::zeros);
    uvec FailCount(NFail + 1, fill::zeros);
    for (size_t i = 0; i < N; i++)
    {
      RiskCount(Y_collapse(i))++;
      FailCount(Y_collapse(i)) += Censor_collapse(i);
    }
    vec RiskCum(NFail + 1, fill::zeros);
    for (size_t j = 0; j <= NFail; j++) RiskCum(j) = RiskCount(j);
    for (size_t j = NFail; j > 0; j--) RiskCum(j-1) += RiskCum(j);
    
    // Weighted score statistic direction
    for (size_t t = 1; t <= NFail; t++)
    {
      if (FailCount(t) == 0) continue;
      
      double wt_at_risk = 0;
      vec wt_x(P, fill::zeros);
      for (size_t i = 0; i < N; i++)
      {
        if (Y_collapse(i) >= t)
        {
          wt_at_risk += node_wt(i);
          wt_x += node_wt(i) * newX.row(i).t();
        }
      }
      
      if (wt_at_risk < 1e-10) continue;
      
      vec mean_x = wt_x / wt_at_risk;
      
      for (size_t i = 0; i < N; i++)
        if (Y_collapse(i) == t && Censor_collapse(i) == 1)
          score_dir += node_wt(i) * (newX.row(i).t() - mean_x);
    }
    
    V.col(0) = score_dir % sqrt(split_vi);
  }
  else
  {
    Surv_CoxPH_Loading(V, newX, Y_collapse, Censor_collapse, split_vi, NFail, P, N,
                        node_wt, useobsweight);
  }
  
  // Check for NaN/Inf
  if (V.has_nan() || V.has_inf())
  {
    V.zeros(P, 1);
    V(0, 0) = 1.0;
  }
  
  // record splitting variable and loading
  OneSplit.var.subvec(0, P-1) = split_var;
  OneSplit.load.subvec(0, P-1) = V.col(0);
  
  // Project data: U1 = newX * v
  arma::vec U1 = newX * V.col(0);
  
  // Check if all U1 values are identical
  if (U1.min() == U1.max())
  {
    OneSplit.score = -1;
    return;
  }
  
  // Find best split on projected variable U1
  // Dispatch on split_rule for cutpoint scoring
  
  // Pre-compute fail/risk counts (needed by all split rules)
  uvec All_Fail_Count(NFail+1, fill::zeros);
  vec All_Risk(NFail+1, fill::zeros);
  
  for (size_t i = 0; i < N; i++)
  {
    All_Risk(Y_collapse(i)) += 1;
    All_Fail_Count(Y_collapse(i)) += Censor_collapse(i);
  }
  
  // manual cumulative sum (vec version)
  for (size_t j = NFail; j > 0; j--)
    All_Risk(j-1) += All_Risk(j);
  
  uvec allid = regspace<uvec>(0, N-1);
  
  Split_Class TempSplit;
  TempSplit.var = 0;
  TempSplit.value = 0;
  TempSplit.score = -1;
  
  if (Param.split_rule == 1)
  {
    // ========== LOGRANK cutpoint ==========
    // Weighted arrays (use node_wt even for "unweighted" — node_wt is all 1.0 if no obs weights)
    vec All_WFail(NFail+1, fill::zeros);
    vec All_WRisk_raw(NFail+1, fill::zeros);
    vec All_W2Risk_raw(NFail+1, fill::zeros);
    
    for (size_t i = 0; i < N; i++)
    {
      double wi = node_wt(i);
      All_WRisk_raw(Y_collapse(i)) += wi;
      All_WFail(Y_collapse(i)) += wi * Censor_collapse(i);
      All_W2Risk_raw(Y_collapse(i)) += wi * wi;
    }
    
    vec All_WRisk(NFail+1, fill::zeros);
    vec All_W2Risk(NFail+1, fill::zeros);
    All_WRisk(NFail) = All_WRisk_raw(NFail);
    All_W2Risk(NFail) = All_W2Risk_raw(NFail);
    for (size_t j = NFail; j > 0; j--)
    {
      All_WRisk(j-1) = All_WRisk_raw(j-1) + All_WRisk(j);
      All_W2Risk(j-1) = All_W2Risk_raw(j-1) + All_W2Risk(j);
    }
    
    uvec All_Risk_Cum = conv_to<uvec>::from(All_Risk);
    
    Surv_Uni_Logrank_Cont(TempSplit,
                          allid,
                          U1,
                          Y_collapse,
                          Censor_collapse,
                          NFail,
                          All_Fail_Count,
                          All_Risk_Cum,
                          Param.nsplit,
                          Param.alpha,
                          rngl,
                          useobsweight,
                          node_wt,
                          All_WFail,
                          All_WRisk,
                          All_W2Risk);
  }
  else if (Param.split_rule == 2)
  {
    // ========== SUPLOGRANK cutpoint ==========
    if (useobsweight)
    {
      // Weighted suplogrank path for LC cutpoint
      vec All_WFail(NFail+1, fill::zeros);
      vec All_WRisk_raw(NFail+1, fill::zeros);
      vec All_W2Risk_raw(NFail+1, fill::zeros);
      vec All_WRisk(NFail+1, fill::zeros);
      vec All_W2Risk(NFail+1, fill::zeros);
      
      for (size_t i = 0; i < N; i++)
      {
        double wi = node_wt(i);
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
      
      Surv_Uni_SupLogrank_Cont(TempSplit,
                               allid,
                               U1,
                               Y_collapse,
                               Censor_collapse,
                               NFail,
                               All_Fail_Count,
                               All_Risk,
                               Param.nsplit,
                               Param.alpha,
                               rngl,
                               node_wt,
                               All_WFail,
                               All_WRisk,
                               All_W2Risk);
    }
    else
    {
      // Unweighted suplogrank
      vec Temp_Vec(NFail+1, fill::zeros);
      vec fail = conv_to<vec>::from(All_Fail_Count);
      Temp_Vec = (1.0 - (fail - 1.0) / (All_Risk - 1.0)) % fail / All_Risk;
      Temp_Vec.elem(find(All_Risk < 2)).zeros();
      
      Surv_Uni_Split_Cont(TempSplit,
                          allid,
                          U1,
                          Y_collapse,
                          Censor_collapse,
                          NFail,
                          All_Fail_Count,
                          All_Risk,
                          Temp_Vec,
                          SURV_DATA.obsweight,
                          Param.split_rule,
                          Param.nsplit,
                          Param.alpha,
                          Param.useobsweight,
                          rngl);
    }
  }
  else if (Param.split_rule == 3)
  {
    // ========== COXGRAD cutpoint ==========
    // Compute pseudo-outcome z_eta for the projected variable U1
    // z_eta = delta_i - cumhaz(t_i) * exp(U1_i * beta)
    // For LC cutpoint, we use the marginal CoxGrad approach on U1
    
    vec z_eta(N, fill::zeros);
    
    // Compute baseline cumulative hazard (Breslow estimator on U1)
    vec risk_sum(NFail + 1, fill::zeros);
    for (size_t i = 0; i < N; i++)
      risk_sum(Y_collapse(i)) += node_wt(i);
    
    // Cumulative from right
    vec risk_cum(NFail + 1, fill::zeros);
    risk_cum(NFail) = risk_sum(NFail);
    for (size_t j = NFail; j > 0; j--)
      risk_cum(j-1) = risk_sum(j-1) + risk_cum(j);
    
    // Baseline hazard at each failure time
    vec base_haz(NFail + 1, fill::zeros);
    vec cumhaz(NFail + 1, fill::zeros);
    for (size_t t = 1; t <= NFail; t++)
    {
      if (All_Fail_Count(t) > 0 && risk_cum(t) > 0)
        base_haz(t) = All_Fail_Count(t) / risk_cum(t);
      cumhaz(t) = cumhaz(t-1) + base_haz(t);
    }
    
    // z_eta = delta_i - H_0(Y_i) for each observation
    for (size_t i = 0; i < N; i++)
    {
      z_eta(i) = Censor_collapse(i) - cumhaz(Y_collapse(i));
      if (useobsweight)
        z_eta(i) *= node_wt(i);
    }
    
    // Now find best cutpoint on U1 using CoxGrad scoring
    // Pseudo-X: binary indicator (left=1)
    if (Param.nsplit > 0)
    {
      // Random cuts
      for (size_t k = 0; k < Param.nsplit; k++)
      {
        double temp_cut = U1(allid(rngl.rand_sizet(0, N-1)));
        uvec Pseudo_X(N, fill::zeros);
        for (size_t i = 0; i < N; i++)
          if (U1(i) <= temp_cut) Pseudo_X(i) = 1;
        
        double temp_score;
        if (useobsweight)
          temp_score = CoxGrad_W(Pseudo_X, z_eta, node_wt);
        else
          temp_score = CoxGrad(Pseudo_X, z_eta);
        
        if (temp_score > TempSplit.score)
        {
          TempSplit.value = temp_cut;
          TempSplit.score = temp_score;
        }
      }
    }
    else
    {
      // Best split: exhaust all cutpoints on sorted U1
      uvec obs_ranked = sort_index(U1);
      uvec indices = allid(obs_ranked);
      
      // Check identical
      if (U1(indices(0)) != U1(indices(N-1)))
      {
        size_t lowindex = 0;
        size_t highindex = N - 2;
        
        if (Param.alpha > 0)
        {
          size_t nmin = (size_t)(N * Param.alpha);
          if (nmin < 1) nmin = 1;
          lowindex = nmin - 1;
          highindex = N - nmin - 1;
        }
        
        // Move past ties at boundaries
        while (lowindex < highindex && U1(indices(lowindex)) == U1(indices(lowindex+1)))
          lowindex++;
        while (highindex > lowindex && U1(indices(highindex)) == U1(indices(highindex+1)))
          highindex--;
        
        if (lowindex <= highindex)
        {
          uvec Pseudo_X(N, fill::zeros);
          // Initialize: left set = indices[0..lowindex]
          for (size_t i = 0; i <= lowindex; i++)
            Pseudo_X(obs_ranked(i)) = 1;
          
          for (size_t i = lowindex; i <= highindex; i++)
          {
            // Skip ties
            if (i > lowindex || i == lowindex)
            {
              while (i < highindex && U1(indices(i)) == U1(indices(i+1)))
              {
                Pseudo_X(obs_ranked(i+1)) = 1;
                i++;
              }
            }
            
            double temp_score;
            if (useobsweight)
              temp_score = CoxGrad_W(Pseudo_X, z_eta, node_wt);
            else
              temp_score = CoxGrad(Pseudo_X, z_eta);
            
            if (temp_score > TempSplit.score)
            {
              TempSplit.value = (U1(indices(i)) + U1(indices(i+1))) / 2.0;
              TempSplit.score = temp_score;
            }
            
            // Add next observation to left set
            if (i + 1 <= highindex)
              Pseudo_X(obs_ranked(i+1)) = 1;
          }
        }
      }
    }
  }
  
  // Safety check
  if (!std::isfinite(TempSplit.score))
    TempSplit.score = -1;
  if (TempSplit.score > 0)
  {
    size_t n_left = sum(U1 <= TempSplit.value);
    if (n_left == 0 || n_left == N)
    {
      TempSplit.score = -1;
      TempSplit.value = 0;
    }
  }
  
  OneSplit.value = TempSplit.value;
  OneSplit.score = TempSplit.score;
}
