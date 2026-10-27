//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Statistical Functions
//  **********************************

// my header file
# include "core/Stat_Function.h"
# include "core/Utility.h"

using namespace Rcpp;
using namespace arma;

// pca
arma::mat xpc(arma::mat& newX,
              arma::vec& newW,
              bool useobsweight)
{
  size_t P = newX.n_cols;
  size_t N = newX.n_rows;
  
  // Edge case: empty matrix or too few observations
  if (N == 0 || P == 0)
  {
    // Return empty matrix
    mat result(P, 1, fill::zeros);
    return result;
  }
  
  if (N < 2)
  {
    // With only 1 observation, can't compute covariance
    // Return first variable as the loading
    mat result(P, 1, fill::zeros);
    result(0, 0) = 1.0;
    return result;
  }
  
  mat X = newX;
  mat C; // covariance matrix
  
  if (useobsweight)
  {
    // Calculate the weighted means
    double sumw = sum(newW);
    
    if (sumw <= 0)
    {
      // Invalid weights, fall back to first variable
      mat result(P, 1, fill::zeros);
      result(0, 0) = 1.0;
      return result;
    }
    
    rowvec wmeans = sum(X.each_col() % newW, 0) / sumw;
    
    // de-mean
    X.each_row() -= wmeans;
    
    // Calculate the weighted standard deviations
    mat XW = X.each_col() % sqrt(newW);
    rowvec wsds = sqrt( sum( square(XW), 0) );
    
    // Handle zero standard deviations (constant columns)
    for (size_t j = 0; j < P; j++)
    {
      if (wsds(j) == 0)
        wsds(j) = 1.0;  // Avoid division by zero
    }
    
    // Standardize the data
    X.each_row() /= wsds;

    // Calculate the weighted covariance matrix
    // X is already multiplied by sqrt(W) at each column before
    C = X.t() * X;
    C = C / sumw;
    
  }else{
    
    // Standardize the data by subtracting the means and dividing by the standard deviations
    rowvec means = mean(X);
    rowvec sds = stddev(X);
    
    // Handle zero standard deviations (constant columns)
    for (size_t j = 0; j < P; j++)
    {
      if (sds(j) == 0)
        sds(j) = 1.0;  // Avoid division by zero
    }
    
    X.each_row() -= means;
    X.each_row() /= sds;
    
    // Calculate the covariance matrix
    C = X.t() * X / (N - 1);
  }
  
  // Calculate the eigenvalues and eigenvectors of the weighted covariance matrix
  vec eigenvalues;
  mat eigenvectors;
  
  // Ensure covariance matrix is symmetric
  C = (C + C.t()) / 2;
  
  try {
    eig_sym(eigenvalues, eigenvectors, C, "std");
  } catch (...) {
    // If eigenvalue decomposition fails, return first variable as loading
    mat result(P, 1, fill::zeros);
    result(0, 0) = 1.0;
    return result;
  }
  
  uvec indices = stable_sort_index(eigenvalues, "descend");
  
  eigenvectors = eigenvectors.cols(indices);

  return(eigenvectors);
}


// sliced inverse regression 
arma::mat sir(arma::mat& newX, 
              arma::vec& newY, 
              arma::vec& newW,
              bool useobsweight,
              size_t nslice)
{
  // prepare objects
  size_t N = newX.n_rows;
  uvec index = stable_sort_index(newY);
  mat X = newX.rows(index);
  vec Y = newY(index);

  // obs index
  size_t slice_size = (size_t) N/nslice;
  size_t res = N - nslice*slice_size; 
  
  // start calculating M
  mat M(newX.n_cols, newX.n_cols, fill::zeros);
  mat Sigma;
  size_t rownum = 0;
  
  if (useobsweight)
  {
    vec W = newW(index);
    double sumw = sum(W);
    
    // apply weight to each row diag(W) * X
    mat XW = X.each_col() % W;
    rowvec XMean = sum(XW, 0) / sumw;
    
    // center X
    mat X_centered = X.each_row() - XMean;
    mat X_C_W = X_centered.each_col() % W;
    
    // cov
    Sigma = X_centered.t() * X_C_W / sumw;    
    
    // calculate M
    size_t positive_slices = 0;
    for (size_t k = 0; k < nslice; k++)
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice mean and weight
      rowvec xhbar = sum(X_C_W.rows(rownum, rownum + nh - 1), 0);
      double wh = accu(W.subvec(rownum, rownum + nh - 1));
      
      // next slice
      rownum += nh;
      
      // add to estimation matrix (adjust for wh)
      if (wh > 0)
      {
        positive_slices++;
        M += xhbar.t() * xhbar / wh;
      }
    }
    if (positive_slices < 2)
      return mat();
  }else{
    // center data
    mat X_centered = X.each_row() - mean(X, 0);
    
    // cov
    Sigma = (1.0 / (N - 1)) * X_centered.t() * X_centered;

    for (size_t k = 0; k < nslice; k++)
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice sum
      rowvec xhbar = sum(X_centered.rows(rownum, rownum + nh - 1), 0);
      
      // next slice
      rownum += nh;
      
      // add to estimation matrix (adjust for wh)
      M += xhbar.t() * xhbar / nh;
    }
  }
  
  Sigma = (Sigma + Sigma.t()) / 2;
  M = (M + M.t()) / 2 / N;
  
  // solve a general eigen problem with
  // M b = lambda C b
  
  // Regularize Sigma to handle singularity
  // Get eigenvalues of Sigma to find smallest positive one
  vec sigma_eigval;
  mat sigma_eigvec;
  if (!Sigma.is_finite() || !M.is_finite() ||
      !eig_sym(sigma_eigval, sigma_eigvec, Sigma))
    return mat();
  
  // Find smallest positive eigenvalue
  double min_pos_eig = 1e-5;  // default floor
  for (size_t i = 0; i < sigma_eigval.n_elem; i++)
  {
    if (sigma_eigval(i) > 0)
    {
      min_pos_eig = sigma_eigval(i);
      break;  // eigenvalues are sorted ascending
    }
  }
  
  // Regularization threshold: min(0.1 * min_positive_eig, 1e-5)
  double ridge = std::min(0.1 * min_pos_eig, 1e-5);
  
  // Replace non-positive eigenvalues with ridge value
  for (size_t i = 0; i < sigma_eigval.n_elem; i++)
  {
    if (sigma_eigval(i) <= 0)
      sigma_eigval(i) = ridge;
  }
  
  // Reconstruct regularized Sigma
  Sigma = sigma_eigvec * diagmat(sigma_eigval) * sigma_eigvec.t();
  Sigma = (Sigma + Sigma.t()) / 2;  // ensure symmetry
  
  // Perform Cholesky decomposition on regularized Sigma
  // Add try-catch for safety
  mat L;
  try {
    L = chol(Sigma, "lower");
  } catch (...) {
    // Fallback: use first variable (simple approach)
    mat result(newX.n_cols, 1, fill::zeros);
    result(0, 0) = 1.0;
    return result;
  }
  
  // Invert the lower triangular matrix L
  mat L_inv = inv(L);
  
  // Transform the problem into a standard eigenvalue problem
  arma::mat M_transformed = L_inv.t() * M * L_inv;
  
  // we use the M b = lambda Sigma b approach
  vec eigval;
  mat eigvec;
  
  if (!M_transformed.is_finite() ||
      !eig_sym(eigval, eigvec, M_transformed) || !eigvec.is_finite())
    return mat();
  
  uvec indices = stable_sort_index(eigval, "descend");
  
  return eigvec.cols(indices);
}

