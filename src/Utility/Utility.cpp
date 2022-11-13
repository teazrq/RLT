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

// debug function

void printLog(const char* mode, const char* x, const int n1, const double n2)
{
  FILE* pFile = fopen("RLT_Debug_log.txt", mode);

  if(pFile != NULL)
    fprintf(pFile, x, n1, n2);

  fclose(pFile);
  return;
}

// generate random integer in [min, max]
/*
int intRand(const int & min, const int & max) {
  static thread_local std::mt19937 generator;
  std::uniform_int_distribution<int> distribution(min, max); 
  return distribution(generator);
}
*/
// simple math

template <class T> const T& max(const T& a, const T& b) {
  return (a<b)?b:a;
}

template <class T> const T& min(const T& a, const T& b) {
  return (a<b)?a:b;
}

// vector in-place reverse cumsum
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




