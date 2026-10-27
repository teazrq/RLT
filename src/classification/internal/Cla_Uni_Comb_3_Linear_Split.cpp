//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification - Linear Combination Split
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;


// ========================================================================
// Method 1: LDA Loading (default)
// Linear Discriminant Analysis directions
// For K classes: up to min(K-1, P) directions
// ========================================================================
void Cla_LDA_Loading(mat& V,
                      const mat& newX,
                      const uvec& newY,
                      const vec& split_vi,
                      const vec& obs_weight,
                      bool useobsweight,
                      size_t nclass,
                      size_t P,
                      size_t N)
{
  // Center each variable
  mat X_centered = newX;
  for (size_t j = 0; j < P; j++)
    X_centered.col(j) = newX.col(j) - mean(newX.col(j));
  
  // Compute pooled within-class scatter matrix W
  mat W(P, P, fill::zeros);
  
  if (useobsweight)
  {
    for (size_t k = 0; k < nclass; k++)
    {
      uvec class_k_idx = find(newY == k);
      size_t n_k = class_k_idx.n_elem;
      
      if (n_k > 1)
      {
        mat X_k = X_centered.rows(class_k_idx);
        vec w_k = obs_weight(class_k_idx);
        double w_sum = sum(w_k);
        mat XkW = X_k.each_col() % w_k;
        vec mu_k = sum(XkW, 0).t() / w_sum;
        mat X_k_centered = X_k.each_row() - mu_k.t();
        // Weighted scatter: X_k' diag(w_k) X_k
        mat Xw = X_k_centered.each_col() % sqrt(w_k);
        W += Xw.t() * Xw;
      }
    }
  }else{
    for (size_t k = 0; k < nclass; k++)
    {
      uvec class_k_idx = find(newY == k);
      size_t n_k = class_k_idx.n_elem;
      
      if (n_k > 1)
      {
        mat X_k = X_centered.rows(class_k_idx);
        vec mu_k = mean(X_k, 0).t();
        mat X_k_centered = X_k.each_row() - mu_k.t();
        W += X_k_centered.t() * X_k_centered;
      }
    }
  }
  
  if (nclass == 2)
  {
    // Binary: single direction W^{-1}(mu1 - mu0)
    uvec class0_idx = find(newY == 0);
    uvec class1_idx = find(newY == 1);
    
    V.set_size(P, 1);
    V.zeros();
    
    if (class0_idx.n_elem > 0 && class1_idx.n_elem > 0)
    {
      vec mu0, mu1;
      
      if (useobsweight)
      {
        double w0 = sum(obs_weight(class0_idx));
        double w1 = sum(obs_weight(class1_idx));
        mat X0 = newX.rows(class0_idx); mat X0W = X0.each_col() % obs_weight(class0_idx);
        mat X1 = newX.rows(class1_idx); mat X1W = X1.each_col() % obs_weight(class1_idx);
        mu0 = sum(X0W, 0).t() / w0;
        mu1 = sum(X1W, 0).t() / w1;
        mu0 -= mean(newX, 0).t();
        mu1 -= mean(newX, 0).t();
      }else{
        mu0 = mean(X_centered.rows(class0_idx), 0).t();
        mu1 = mean(X_centered.rows(class1_idx), 0).t();
      }
      vec delta = mu1 - mu0;
      
      W.diag() += 1e-6 * trace(W) / P;
      
      try {
        vec direction = pinv(W) * delta;
        
        double norm_dir = norm(direction, 2);
        if (norm_dir > 0)
          direction = direction / norm_dir;
        
        uword max_idx = index_max(abs(direction));
        if (direction(max_idx) < 0)
          direction = -direction;
        
        V.col(0) = direction;
        return;
      } catch (...) {
        // pinv failed: fall through to fallback
      }
    }
    
    V(0, 0) = 1.0;
    return;
  }
  
  // Multi-class: eigenvectors of W^{-1}B with positive eigenvalues
  vec grand_mean = mean(X_centered, 0).t();
  
  mat B(P, P, fill::zeros);
  if (useobsweight)
  {
    for (size_t k = 0; k < nclass; k++)
    {
      uvec class_k_idx = find(newY == k);
      
      if (class_k_idx.n_elem > 0)
      {
        vec w_k = obs_weight(class_k_idx);
        double w_sum = sum(w_k);
        mat Xk = X_centered.rows(class_k_idx);
        vec mu_k = sum(Xk.each_col() % w_k, 0).t() / w_sum;
        vec diff = mu_k - grand_mean;
        B += w_sum * (diff * diff.t());
      }
    }
  }else{
    for (size_t k = 0; k < nclass; k++)
    {
      uvec class_k_idx = find(newY == k);
      size_t n_k = class_k_idx.n_elem;
      
      if (n_k > 0)
      {
        vec mu_k = mean(X_centered.rows(class_k_idx), 0).t();
        vec diff = mu_k - grand_mean;
        B += (double)n_k * (diff * diff.t());
      }
    }
  }
  
  W.diag() += 1e-4 * trace(W) / P;
  
  try {
    mat Winv_B = pinv(W) * B;
    
    cx_vec eigvals;
    cx_mat eigvecs;
    eig_gen(eigvals, eigvecs, Winv_B);
    
    vec real_eigvals = real(eigvals);
    mat real_eigvecs = real(eigvecs);
    
    // Sort eigenvalues descending
    uvec sorted_idx = stable_sort_index(real_eigvals, "descend");
    
    // Count directions with positive eigenvalues
    size_t D = 0;
    for (size_t i = 0; i < sorted_idx.n_elem; i++)
    {
      if (real_eigvals(sorted_idx(i)) > 0)
        D++;
      else
        break;
    }
    
    size_t max_D = std::min((size_t)(nclass - 1), P);
    D = std::min(D, max_D);
    
    if (D > 0)
    {
      V.set_size(P, D);
      
      for (size_t d = 0; d < D; d++)
      {
        vec dir = real_eigvecs.col(sorted_idx(d));
        
        double norm_dir = norm(dir, 2);
        if (norm_dir > 0)
          dir = dir / norm_dir;
        
        uword max_idx = index_max(abs(dir));
        if (dir(max_idx) < 0)
          dir = -dir;
        
        V.col(d) = dir;
      }
      return;
    }
  } catch (...) {
    // eig_gen/pinv failed: fall through to fallback
  }
  
  // Fallback
  V.set_size(P, 1);
  V.zeros();
  V(0, 0) = 1.0;
}


// ========================================================================
// Method 2: Naive Loading (formerly CVGate)
// SVD of F-stat weighted class means with sqrt(VI) weighting.
// No internal gating — relies on outer embed.threshold for variable filtering.
// ========================================================================
void Cla_Naive_Loading(mat& V,
                        const mat& newX,
                        const uvec& newY,
                        const vec& split_vi,
                        const vec& obs_weight,
                        bool useobsweight,
                        size_t nclass,
                        size_t P,
                        size_t N)
{
  // Step 1: Compute F-statistics
  vec F_stats(P, fill::zeros);
  
  if (useobsweight)
  {
    double total_w = sum(obs_weight);
    
    for (size_t j = 0; j < P; j++)
    {
      vec x_j = newX.col(j);
      double wgrand_mean = sum(obs_weight % x_j) / total_w;
      
      double SS_between = 0.0;
      double SS_within = 0.0;
      
      for (size_t k = 0; k < nclass; k++)
      {
        uvec class_k_idx = find(newY == k);
        
        if (class_k_idx.n_elem > 0)
        {
          vec w_k = obs_weight(class_k_idx);
          double w_sum = sum(w_k);
          double class_mean = sum(w_k % x_j(class_k_idx)) / w_sum;
          SS_between += w_sum * pow(class_mean - wgrand_mean, 2);
          
          vec deviations = x_j(class_k_idx) - class_mean;
          SS_within += sum(w_k % square(deviations));
        }
      }
      
      if (SS_within > 0 && nclass > 1)
      {
        double df_between = nclass - 1;
        double df_within = total_w - nclass;
        
        if (df_within > 0)
          F_stats(j) = (SS_between / df_between) / (SS_within / df_within);
      }
    }
  }else{
    for (size_t j = 0; j < P; j++)
    {
      vec x_j = newX.col(j);
      double grand_mean = mean(x_j);
      
      double SS_between = 0.0;
      double SS_within = 0.0;
      
      for (size_t k = 0; k < nclass; k++)
      {
        uvec class_k_idx = find(newY == k);
        size_t n_k = class_k_idx.n_elem;
        
        if (n_k > 0)
        {
          double class_mean = mean(x_j(class_k_idx));
          SS_between += n_k * pow(class_mean - grand_mean, 2);
          
          vec deviations = x_j(class_k_idx) - class_mean;
          SS_within += sum(deviations % deviations);
        }
      }
      
      if (SS_within > 0 && nclass > 1)
      {
        double df_between = nclass - 1;
        double df_within = N - nclass;
        
        if (df_within > 0)
          F_stats(j) = (SS_between / df_between) / (SS_within / df_within);
      }
    }
  }
  
  // Step 2: SVD of centered class means
  mat X_centered = newX;
  for (size_t j = 0; j < P; j++)
    X_centered.col(j) = newX.col(j) - mean(newX.col(j));
  
  mat M(nclass, P, fill::zeros);
  
  if (useobsweight)
  {
    for (size_t k = 0; k < nclass; k++)
    {
      uvec class_k_idx = find(newY == k);
      if (class_k_idx.n_elem > 0)
      {
        vec w_k = obs_weight(class_k_idx);
        double w_sum = sum(w_k);
        mat Xc = X_centered.rows(class_k_idx);
        for (size_t j = 0; j < P; j++)
        {
          M(k, j) = sum(w_k % Xc.col(j)) / w_sum;
        }
      }
    }
  }else{
    for (size_t k = 0; k < nclass; k++)
    {
      uvec class_k_idx = find(newY == k);
      if (class_k_idx.n_elem > 0)
      {
        for (size_t j = 0; j < P; j++)
        {
          vec x_j_centered = X_centered.col(j);
          M(k, j) = mean(x_j_centered(class_k_idx));
        }
      }
    }
  }
  
  // Apply F-stat weighting to columns of class means matrix
  for (size_t j = 0; j < P; j++)
    M.col(j) = M.col(j) * sqrt(F_stats(j) / (sum(F_stats) + 1e-10));
  
  size_t D = std::min((size_t)nclass, (size_t)P);
  
  if (nclass >= 2 && P >= 2)
  {
    try {
      mat U, Vsvd;
      vec S;
      
      if (svd(U, S, Vsvd, M) && Vsvd.n_cols > 0)
      {
        size_t D_actual = std::min(D, (size_t)Vsvd.n_cols);
        V.set_size(P, D_actual);
        
        for (size_t d = 0; d < D_actual; d++)
        {
          vec dir = Vsvd.col(d);
          
          uword max_idx = index_max(abs(dir));
          if (dir(max_idx) < 0)
            dir = -dir;
          
          // Apply sqrt(VI) weighting (standard approach)
          dir = dir % sqrt(split_vi);
          
          double norm_dir = norm(dir, 2);
          if (norm_dir > 0)
            dir = dir / norm_dir;
          
          V.col(d) = dir;
        }
        return;
      }
    } catch (...) {
      // svd failed: fall through to fallback
    }
  }
  
  // Fallback
  V.set_size(P, 1);
  V.zeros();
  V(0, 0) = 1.0;
}


// ========================================================================
// Method 3: Random Projection with Class-Mean Tilt
// Generate random directions tilted toward class separation
// Very cheap: no matrix inversions or decompositions.
// ========================================================================
void Cla_Random_Loading(mat& V,
                         const mat& newX,
                         const uvec& newY,
                         const vec& split_vi,
                         const vec& obs_weight,
                         bool useobsweight,
                         size_t nclass,
                         size_t P,
                         size_t N)
{
  // Compute class means
  mat X_centered = newX;
  for (size_t j = 0; j < P; j++)
    X_centered.col(j) = newX.col(j) - mean(newX.col(j));
  
  // Overall separation direction: weighted sum of class mean differences
  vec sep_dir(P, fill::zeros);
  vec grand_mean = mean(X_centered, 0).t();
  
  if (useobsweight)
  {
    for (size_t k = 0; k < nclass; k++)
    {
      uvec idx = find(newY == k);
      if (idx.n_elem > 0)
      {
        vec w_k = obs_weight(idx);
        double w_sum = sum(w_k);
        mat Xidx = X_centered.rows(idx);
        vec mu_k = sum(Xidx.each_col() % w_k, 0).t() / w_sum;
        sep_dir += w_sum * (mu_k - grand_mean);
      }
    }
  }else{
    for (size_t k = 0; k < nclass; k++)
    {
      uvec idx = find(newY == k);
      if (idx.n_elem > 0)
      {
        vec mu_k = mean(X_centered.rows(idx), 0).t();
        sep_dir += (double)idx.n_elem * (mu_k - grand_mean);
      }
    }
  }
  
  double norm_sep = norm(sep_dir, 2);
  if (norm_sep > 0)
    sep_dir = sep_dir / norm_sep;
  
  // Generate R tilted random directions
  size_t R = std::min((size_t)16, nclass * (nclass - 1) / 2 + P);
  size_t n_keep = std::min((size_t)(nclass - 1), P);
  if (n_keep < 1) n_keep = 1;
  
  mat candidates(P, R, fill::zeros);
  vec quality(R, fill::zeros);
  size_t r_count = 0;
  
  // Direction set 1: class-pair mean differences (deterministic, data-driven)
  for (size_t i = 0; i < nclass && r_count < R; i++)
  {
    uvec idx_i = find(newY == i);
    if (idx_i.n_elem == 0) continue;
    
    vec mu_i;
    if (useobsweight)
    {
      vec w_i = obs_weight(idx_i);
      mat Xi = X_centered.rows(idx_i);
      mu_i = sum(Xi.each_col() % w_i, 0).t() / sum(w_i);
    }else{
      mu_i = mean(X_centered.rows(idx_i), 0).t();
    }
    
    for (size_t j = i + 1; j < nclass && r_count < R; j++)
    {
      uvec idx_j = find(newY == j);
      if (idx_j.n_elem == 0) continue;
      
      vec mu_j;
      if (useobsweight)
      {
        vec w_j = obs_weight(idx_j);
        mat Xj = X_centered.rows(idx_j);
        mu_j = sum(Xj.each_col() % w_j, 0).t() / sum(w_j);
      }else{
        mu_j = mean(X_centered.rows(idx_j), 0).t();
      }
      
      vec dir = mu_j - mu_i;
      dir = 0.5 * dir + 0.5 * sep_dir;  // tilt toward overall separation
      double norm_d = norm(dir, 2);
      if (norm_d > 0) dir = dir / norm_d;
      candidates.col(r_count) = dir % sqrt(split_vi);
      r_count++;
    }
  }
  
  // Direction set 2: coordinate projections tilted toward sep_dir
  for (size_t j = 0; j < P && r_count < R; j++)
  {
    vec dir(P, fill::zeros);
    dir(j) = 1.0;
    dir = 0.3 * sep_dir + 0.7 * dir;
    double norm_d = norm(dir, 2);
    if (norm_d > 0) dir = dir / norm_d;
    candidates.col(r_count) = dir % sqrt(split_vi);
    r_count++;
  }
  
  R = r_count;  // actual number of candidates
  
  // Evaluate quality of each candidate direction
  for (size_t r = 0; r < R; r++)
  {
    vec tilted = candidates.col(r);
    
    // Quality measure: between-class variance on projection
    vec proj = X_centered * tilted;
    double total_mean = mean(proj);
    double ss_between = 0;
    
    if (useobsweight)
    {
      double w_total = sum(obs_weight);
      double wgrand_mean = sum(obs_weight % proj) / w_total;
      
      for (size_t k = 0; k < nclass; k++)
      {
        uvec idx = find(newY == k);
        if (idx.n_elem > 0)
        {
          vec w_k = obs_weight(idx);
          double w_sum = sum(w_k);
          double class_mean = sum(w_k % proj(idx)) / w_sum;
          ss_between += w_sum * pow(class_mean - wgrand_mean, 2);
        }
      }
    }else{
      for (size_t k = 0; k < nclass; k++)
      {
        uvec idx = find(newY == k);
        if (idx.n_elem > 0)
        {
          double class_mean = mean(proj(idx));
          ss_between += (double)idx.n_elem * pow(class_mean - total_mean, 2);
        }
      }
    }
    quality(r) = ss_between;
  }
  
  // Keep top n_keep directions by quality
  uvec sorted = stable_sort_index(quality, "descend");
  
  V.set_size(P, n_keep);
  for (size_t d = 0; d < n_keep; d++)
  {
    vec dir = candidates.col(sorted(d));
    double norm_d = norm(dir, 2);
    if (norm_d > 0)
      dir = dir / norm_d;
    
    uword max_idx = index_max(abs(dir));
    if (dir(max_idx) < 0)
      dir = -dir;
    
    V.col(d) = dir;
  }
}


// ========================================================================
// Method 4: Logistic Loading
// SVD of one-vs-all logistic regression coefficients
// Returns V (P x D) with D = min(nclass, P) directions
// ========================================================================
void Cla_Logistic_Loading(mat& V,
                           const mat& newX,
                           const uvec& newY,
                           const vec& split_vi,
                           const vec& obs_weight,
                           bool useobsweight,
                           size_t nclass,
                           size_t P,
                           size_t N)
{
  // Center each variable
  mat X_centered = newX;
  for (size_t j = 0; j < P; j++)
    X_centered.col(j) = newX.col(j) - mean(newX.col(j));
  
  // Build coefficient matrix M (nclass x P) via one-vs-all logistic regression
  mat M(nclass, P, fill::zeros);
  
  for (size_t k = 0; k < nclass; k++)
  {
    vec y_binary(N, fill::zeros);
    uvec class_k_idx = find(newY == k);
    y_binary(class_k_idx).ones();
    
    // IRLS
    vec beta(P, fill::zeros);
    
    for (int iter = 0; iter < 10; iter++)
    {
      vec eta = X_centered * beta;
      vec p = 1.0 / (1.0 + exp(-eta));
      p = clamp(p, 1e-10, 1.0 - 1e-10);
      
      vec w = p % (1.0 - p);
      vec z = eta + (y_binary - p) / w;
      
      if (useobsweight)
        w = w % obs_weight;  // Combine IRLS weights with observation weights
      
      mat Xw = X_centered;
      for (size_t j = 0; j < P; j++)
        Xw.col(j) = Xw.col(j) % sqrt(w);
      
      vec z_w = z % sqrt(w);
      
      mat XtX = Xw.t() * Xw;
      vec Xtz = Xw.t() * z_w;
      
      XtX.diag() += 1e-6;
      
      try {
        vec beta_new = pinv(XtX) * Xtz;
        if (beta_new.has_nan() || beta_new.has_inf())
          break;
        beta = beta_new;
      } catch (...) {
        break;
      }
    }
    
    M.row(k) = beta.t();
  }
  
  // SVD: take up to min(nclass, P) directions
  size_t D = std::min((size_t)nclass, (size_t)P);
  
  if (nclass >= 2 && P >= 2)
  {
    try {
      mat U, Vsvd;
      vec S;
      
      if (svd(U, S, Vsvd, M) && Vsvd.n_cols > 0)
      {
        size_t D_actual = std::min(D, (size_t)Vsvd.n_cols);
        V.set_size(P, D_actual);
        
        for (size_t d = 0; d < D_actual; d++)
        {
          vec dir = Vsvd.col(d);
          
          uword max_idx = index_max(abs(dir));
          if (dir(max_idx) < 0)
            dir = -dir;
          
          V.col(d) = dir;
        }
        return;
      }
    } catch (...) {
      // svd failed: fall through to fallback
    }
  }
  
  // Fallback
  V.set_size(P, 1);
  V.zeros();
  V(0, 0) = 1.0;
}


// ========================================================================
// Main dispatcher: call the appropriate loading function, loop over
// directions, find the best (direction, cut point) pair
//
// linear_comb_method mapping:
//   1 = LDA (default)
//   2 = Naive (formerly CVGate)
//   3 = Random (formerly RandomProj)
//   4 = Logistic
// ========================================================================
void Cla_Uni_Comb_Linear(Comb_Split_Class& OneSplit,
                         const uvec& split_var,
                         const vec& split_vi,
                         const RLT_CLA_DATA& CLA_DATA,
                         const PARAM_GLOBAL& Param,
                         const uvec& obs_id,
                         Rand& rngl)
{
  bool useobsweight = Param.useobsweight;
  size_t N = obs_id.n_elem;
  size_t P = split_var.n_elem;
  size_t nclass = CLA_DATA.nclass;
  size_t linear_comb_method = Param.linear_comb_method;
  size_t nsplit = Param.nsplit;
  double alpha = Param.alpha;
  
  // Edge case
  if (N < 2 || P == 0)
  {
    OneSplit.var(0) = split_var.n_elem > 0 ? split_var(0) : 0;
    OneSplit.load(0) = 1.0;
    OneSplit.value = 0;
    OneSplit.score = -1;
    return;
  }
  
  // Extract node data
  mat newX(CLA_DATA.X(obs_id, split_var));
  uvec newY = CLA_DATA.Y(obs_id);
  vec newW;
  if (useobsweight) newW = CLA_DATA.obsweight(obs_id);
  
  // Compute loading directions using the appropriate method
  mat V(P, 1, fill::zeros);
  if (linear_comb_method == 1)
    Cla_LDA_Loading(V, newX, newY, split_vi, newW, useobsweight, nclass, P, N);
  else if (linear_comb_method == 2)
    Cla_Naive_Loading(V, newX, newY, split_vi, newW, useobsweight, nclass, P, N);
  else if (linear_comb_method == 3)
    Cla_Random_Loading(V, newX, newY, split_vi, newW, useobsweight, nclass, P, N);
  else if (linear_comb_method == 4)
    Cla_Logistic_Loading(V, newX, newY, split_vi, newW, useobsweight, nclass, P, N);
  
  size_t D = V.n_cols;
  
  // Safety: check for NaN/Inf in any direction
  if (V.has_nan() || V.has_inf())
  {
    V.set_size(P, 1);
    V.zeros();
    V(0, 0) = 1.0;
    D = 1;
  }
  
  // Loop over all directions, find the best (direction, cut point) pair
  double best_score = -1;
  vec best_load(P, fill::zeros);
  double best_value = 0;
  
  uvec allid = regspace<uvec>(0, N-1);
  
  for (size_t d = 0; d < D; d++)
  {
    vec v = V.col(d);
    
    // Skip zero directions
    if (abs(v).max() == 0)
      continue;
    
    // Project data onto this direction
    arma::vec U1 = newX * v;
    
    // Skip if all projections are identical
    if (U1.min() == U1.max())
      continue;
    
    // Find best split point on this projection
    Split_Class TempSplit;
    TempSplit.var = 0;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    Cla_Uni_Split_Cont(TempSplit,
                       allid,
                       U1,
                       newY,
                       newW,
                       nclass,
                       0.0,
                       1,  // Gini impurity
                       nsplit,
                       alpha,
                       useobsweight,
                       rngl);
    
    if (TempSplit.score > best_score)
    {
      best_score = TempSplit.score;
      best_load = v;
      best_value = TempSplit.value;
    }
  }
  
  // Record the best split found across all directions
  OneSplit.var.subvec(0, P-1) = split_var;
  OneSplit.load.subvec(0, P-1) = best_load;
  OneSplit.value = best_value;
  OneSplit.score = best_score;
}
