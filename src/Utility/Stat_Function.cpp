//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Statistical Functions
//  **********************************

// my header file
# include "Stat_Function.h"
# include "Utility.h"

using namespace Rcpp;
using namespace arma;

// pca
arma::mat xpc(arma::mat& newX,
              arma::vec& newW,
              bool useobsweight)
{
  //size_t P = newX.n_cols;
  size_t N = newX.n_rows;
  mat X = newX;
  mat C; // covariance matrix
  
  if (useobsweight)
  {
    // Calculate the weighted means
    double sumw = sum(newW);
    rowvec wmeans = sum(X.each_col() % newW, 0) / sumw;
    
    // de-mean
    X.each_row() -= wmeans;
    
    // Calculate the weighted standard deviations
    mat XW = X.each_col() % sqrt(newW);
    rowvec wsds = sqrt( sum( square(XW), 0) );
    
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
    X.each_row() -= means;
    X.each_row() /= sds;
    
    // Calculate the covariance matrix
    C = X.t() * X / (N - 1);
  }
  
  // Calculate the eigenvalues and eigenvectors of the weighted covariance matrix
  vec eigenvalues;
  mat eigenvectors;
  
  eig_sym(eigenvalues, eigenvectors, C, "std");
  
  uvec indices = sort_index(eigenvalues, "descend");
  
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
  uvec index = sort_index(newY);    
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
      M += xhbar.t() * xhbar / wh;
    }
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
  
  // Perform Cholesky decomposition on Sigma
  // Invert the lower triangular matrix L
  mat L_inv = inv(chol(Sigma, "lower"));
  
  // Transform the problem into a standard eigenvalue problem
  arma::mat M_transformed = L_inv.t() * M * L_inv;
  
  // we use the M b = lambda Sigma b approach
  vec eigval;
  mat eigvec;
  
  eig_sym(eigval, eigvec, M_transformed);
  
  uvec indices = sort_index(eigval, "descend");
  
  return eigvec.cols(indices);
}

// save
arma::mat save(arma::mat& newX, 
               arma::vec& newY, 
               arma::vec& newW,
               bool useobsweight,
               size_t nslice)
{
  uvec index = sort_index(newY);
  size_t N = newY.n_elem;
  size_t P = newX.n_cols;
  
  mat M(P, P, fill::zeros);
  
  mat sortedx = newX.rows(index);
  vec sortedy = newY(index);
  
  size_t slice_size = (size_t) N/nslice;
  size_t res = N - nslice*slice_size;
  mat Diag(P, P, fill::eye);
  
  if (useobsweight)
  {
    
    // obs index
    size_t rownum = 0;
    
    for (size_t k =0; k < nslice; k++) 
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice data
      mat xslice = sortedx.rows(rownum, rownum + nh - 1);
      vec wslice = newW.subvec(rownum, rownum + nh - 1);
      
      // next slice
      rownum += nh;
      
      // center xslice
      xslice.each_row() -= mean(xslice.each_col() % wslice, 0); 
      
      // apply weight
      xslice.each_col() %= sqrt(wslice);
      
      // slice cov and M
      double w = sum(wslice);
      w = 1 - norm(wslice, "fro") / w / w;
      
      mat C = Diag - xslice.t() * xslice / w;
      M += C * C.t();
    }
    
    
  }else{
    // obs index
    size_t rownum = 0;
    
    for (size_t k =0; k < nslice; k++) 
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice data
      mat xslice = sortedx.rows(rownum, rownum + nh - 1);
      
      // next slice
      rownum += nh;
      
      RLTcout << "k= " << k << std::endl;
      
      // slice cov and M
      mat C = Diag - arma::cov(xslice);
      M += C * C.t();
      
      RLTcout << "complete " << k << std::endl;
    }
  }
  
  return M;
  
}

