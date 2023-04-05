//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Utility Functions
//  **********************************

// my header file
# include "Utility.h"

// check cores

size_t checkCores(size_t usecores, size_t verbose)
{
  size_t use_cores = ( (usecores == 0) ? omp_get_max_threads():usecores);

  if (use_cores > 1) OMPMSG(1);

  if(use_cores > (size_t) omp_get_max_threads())
  {
    if (verbose) Rprintf("Do not have %i cores, use maximum %i cores. \n", use_cores, omp_get_max_threads());
    use_cores = omp_get_max_threads();
  }
  
  return(use_cores);
}

// debug function to print into a .txt file
void printLog(const char* mode, const char* x, const int n1, const double n2)
{
  FILE* pFile = fopen("RLT_Debug_log.txt", mode);

  if(pFile != NULL)
    fprintf(pFile, x, n1, n2);

  fclose(pFile);
  return;
}

// vector *** in-place *** reverse cumsum
void cumsum_rev(arma::uvec& seq)
{
  // cumulative at risk counts for left
  size_t N = seq.n_elem;
  
  if (N <= 1)
    return;
  
  for (size_t i = N-2; i>0; i--)
    seq(i) += seq(i+1);
  
  seq(0) += seq(1);
}

// ****************//
// field functions //
// ****************//

void field_vec_resize(arma::field<arma::vec>& A, size_t size)
{
  arma::field<arma::vec> B(size);
  
  size_t common_size = (A.n_elem > size) ? size : A.n_elem;
  
  for (size_t i = 0; i < common_size; i++)
  {
    //Was false, true. Triggered an error with new version of RcppArmadillo
    //Less efficient than false, true but works
    B[i] = vec(A[i].begin(), A[i].size(), true, false);
  }
  
  A.set_size(size);
  for (size_t i = 0; i < common_size; i++)
  {
    //Was false, true. Triggered an error with new version of RcppArmadillo
    //Less efficient than false, true but works
    A[i] = vec(B[i].begin(), B[i].size(), true, false);
  }
}

void field_vec_resize(arma::field<arma::uvec>& A, size_t size)
{
  arma::field<arma::uvec> B(size);
  
  size_t common_size = (A.n_elem > size) ? size : A.n_elem;
  
  for (size_t i = 0; i < common_size; i++)
  {
    //Was false, true. Triggered an error with new version of RcppArmadillo
    //Less efficient than false, true but works
    B[i] = uvec(A[i].begin(), A[i].size(), true, false);
  }
  
  A.set_size(size);
  for (size_t i = 0; i < common_size; i++)
  {
    //Was false, true. Triggered an error with new version of RcppArmadillo
    //Less efficient than false, true but works
    A[i] = uvec(B[i].begin(), B[i].size(), true, false);
  }
}



