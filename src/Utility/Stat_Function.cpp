//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Statistical Functions
//  **********************************

// my header file
# include "Stat_Function.h"
# include "Utility.h"

using namespace Rcpp;
using namespace arma;

arma::mat first_pc(arma::mat& newX,
                   arma::vec& newW,
                   bool useobsweight)
{
  size_t P = newX.n_cols;
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
    // do a little trick here
    X.each_col() %= sqrt(newW);
    
    rowvec wsds = sqrt(sum(X, 0)) / sqrt(sumw);
    
    // Standardize the data
    X.each_row() /= wsds;

    // Calculate the weighted covariance matrix
    // X is already multiplied by sqrt(W) at each column before
    C = X.t() * X;
    C = C / accu(newW);
    
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
  //eigenvalues = eigenvalues(indices);
  
  //RLTcout << "pca eigenvalues \n" << eigenvalues << std::endl;
  //RLTcout << "pca eigenvectors \n" << eigenvectors << std::endl;
  
  return(eigenvectors);
}


// sliced inverse regression 

arma::mat sir(arma::mat& newX, 
              arma::vec& newY, 
              arma::vec& newW,
              bool useobsweight,
              size_t nslice)
{
  uvec index = sort_index(newY);
  
  mat x = newX.rows(index);
  vec y = newY(index);
  
  size_t N = newY.n_elem;
  size_t P = newX.n_cols;  
  
  // initiate 
  mat M(P, P, fill::zeros);
  mat C;
  size_t slice_size = (size_t) N/nslice;
  size_t res = N - nslice*slice_size;  
  
  if (useobsweight)
  {
    // calculate the weighted covariance matrix
    // apply weights
    vec w = newW(index);
    
    // center data
    double sumw = sum(w);
    rowvec wmeans = sum(x.each_col() % w, 0) / sumw;
    x.each_row() -= wmeans;
    
    // Calculate the weighted covariance matrix
    x.each_col() %= w;
    mat C = x.t() * newX.rows(index) / accu(newW);

    // obs index
    size_t rownum = 0;
    
    // start to calculate slice means and add to M
    for (size_t k = 0; k < nslice; k++)
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice weight
      double wh = sum(w.subvec(rownum, rownum + nh - 1));
      
      // weighted slice mean (without dividing by wh)
      rowvec xhbar = sum(x.rows(rownum, rownum + nh - 1), 0);
      
      // next slice
      rownum += nh;
      
      // add to estimation matrix (adjust for wh)
      M += xhbar.t() * xhbar / wh;
    }
    
  }else{
    // x mean
    rowvec xbar = mean(x, 0);
    
    // center x
    x.each_row() -= xbar;
    
    // covariance matrix 
    mat C = x.t() * x / (N - 1);
    
    // obs index
    size_t rownum = 0;
    
    for (size_t k =0; k < nslice; k++) 
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice mean (without dividing by wh)
      rowvec xhbar = mean(x.rows(rownum, rownum + nh - 1), 0);
      
      // next slice
      rownum += nh;
      
      // add to estimation matrix (adjust for wh)
      M += xhbar.t() * xhbar / nh;
    }
  }
  
  // we use the M b = lambda Sigma b approach
  vec eigenvalues;
  mat eigenvectors;
  
  eig_sym(eigenvalues, eigenvectors, inv(C, inv_opts::allow_approx) * M, "std");
  
  uvec indices = sort_index(eigenvalues, "descend");
  
  return eigenvectors.cols(indices);
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

