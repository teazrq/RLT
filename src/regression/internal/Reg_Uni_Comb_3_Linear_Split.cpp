//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split using linear combination
void Reg_Uni_Comb_Linear(Comb_Split_Class& OneSplit,
                         const uvec& split_var,
                         const vec& split_vi,
                         const RLT_REG_DATA& REG_DATA,
                         const PARAM_GLOBAL& Param,
                         const uvec& obs_id,
                         Rand& rngl)
{
  // construct some new data 
  bool useobsweight = Param.useobsweight;
  size_t N = obs_id.n_elem;
  size_t P = split_var.n_elem;
  
  // Edge case: not enough observations
  if (N < 2 || P == 0)
  {
    // Return without split (use first variable with zero loading)
    OneSplit.var(0) = split_var.n_elem > 0 ? split_var(0) : 0;
    OneSplit.load(0) = 1.0;
    OneSplit.value = 0;
    OneSplit.score = -1;
    return;
  }
  
  mat newX(REG_DATA.X(obs_id, split_var));
  vec newY(REG_DATA.Y(obs_id));
  vec newW;
  if (useobsweight) newW = REG_DATA.obsweight(obs_id);
  
  // some parameters
  // linear_comb_method: naive(1), lm(2), pca(3), sir(4)
  size_t linear_comb_method = Param.linear_comb_method;
  size_t nsplit = Param.nsplit;
  double alpha = Param.alpha;
  
  // check splitting rule 1 naive; 2 lm; 3 pca; 4 sir; 
  if (linear_comb_method > 1 and N < 10)
  {
    linear_comb_method = 1; // switch to naive if sample size is too small
  }

  // find splitting rule loading vector 
  vec v;
  
  if (linear_comb_method == 1) // naive
  {
    if (useobsweight)
    {
      // Weighted correlation: sign = sign of weighted covariance
      double wsum = sum(newW);
      mat XW = newX.each_col() % newW;
      vec wXmean = sum(XW, 0).t() / wsum;
      double wYmean = sum(newW % newY) / wsum;
      vec wcov = (newX.each_row() - wXmean.t()).t() * (newW % (newY - wYmean));
      v = sqrt(split_vi) % sign(wcov);
    }else{
      v = sqrt(split_vi) % sign(cor(newX, newY));
    }
  }

  if (linear_comb_method == 2) // lm
  {
    try {
      if (useobsweight)
      {
        mat XW = newX;
        XW.each_col() %= sqrt(newW);
        v = pinv(XW.t() * XW) * (newX.t() * (newW % newY));
      }else{
        v = pinv(newX.t() * newX) * (newX.t() * newY);
      }
    } catch (...) {
      // Fall back to naive if solve fails
      v = sqrt(split_vi) % sign(cor(newX, newY));
    }
  }
  
  if (linear_comb_method == 3) // pca
  {
    try {
      mat pca_result = xpc(newX, newW, useobsweight);
      if (pca_result.n_cols > 0 && pca_result.n_rows > 0)
        v = pca_result.col(0);
      else
      {
        // Fall back to naive if PCA returns empty
        v = sqrt(split_vi) % sign(cor(newX, newY));
      }
    } catch (...) {
      // Fall back to naive if PCA fails
      v = sqrt(split_vi) % sign(cor(newX, newY));
    }
  }  
  
  if (linear_comb_method == 4) // sir
  {
    try {
      mat directions = sir(newX, newY, newW, useobsweight, sqrt(N));
      if (directions.n_rows == P && directions.n_cols > 0)
        v = directions.col(0);
    } catch (...) {
      // Use the same naive fallback as the other direction methods.
    }
    if (v.n_elem != P || !v.is_finite())
      v = sqrt(split_vi) % sign(cor(newX, newY));
  }

  // Check for NaN/Inf in loading vector and fall back to naive if needed
  if (v.has_nan() || v.has_inf())
  {
    v = sqrt(split_vi) % sign(cor(newX, newY));
    
    // If still has NaN (e.g., zero std dev), use first variable
    if (v.has_nan() || v.has_inf())
    {
      v.zeros(P);
      v(0) = 1.0;
    }
  }
  
  // record splitting variable and loading
  OneSplit.var.subvec(0, P-1) = split_var;
  OneSplit.load.subvec(0, P-1) = v;
  
  // search for the best splitting point with the linear combination
  arma::vec U1 = newX * v;
  
  // Check if all U1 values are identical (cannot split)
  if (U1.min() == U1.max())
  {
    // All linear combination values are the same, cannot find a valid split
    OneSplit.score = -1;
    return;
  }
  
  //Initialize objects
  Split_Class TempSplit;
  TempSplit.var = 0;
  TempSplit.value = 0;
  TempSplit.score = -1;
  
  uvec allid = regspace<uvec>(0, N-1);
  
  Reg_Uni_Split_Cont(TempSplit,
                     allid,
                     U1,
                     newY,
                     newW,
                     0.0, // penalty
                     1, // univariate splitting rule (var)
                     nsplit,
                     alpha,
                     useobsweight,
                     rngl);
  
  // Safety check: ensure split produces non-empty children
  if (TempSplit.score > 0)
  {
    size_t n_left = sum(U1 <= TempSplit.value);
    if (n_left == 0 || n_left == N)
    {
      // This should not happen, but mark as invalid if it does
      TempSplit.score = -1;
      TempSplit.value = 0;
    }
  }
  
  OneSplit.value = TempSplit.value;
  OneSplit.score = TempSplit.score;
}


