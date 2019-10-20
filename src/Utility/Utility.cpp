//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Utility Functions: check
//  **********************************

// my header file
# include "Utility.h"

// check cores

int checkCores(int usecores, int verbose)
{
  int use_cores = ( (usecores > 1) ? usecores:1 );

  if (use_cores > 0) OMPMSG(1);

  int haveCores = omp_get_max_threads();

  if(use_cores > haveCores)
  {
    if (verbose) Rprintf("Do not have %i cores, use maximum %i cores. \n", use_cores, haveCores);
    use_cores = haveCores;
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

int intRand(const int & min, const int & max) {
  static thread_local std::mt19937 generator;
  std::uniform_int_distribution<int> distribution(min, max);
  return distribution(generator);
}

// simple math

template <class T> const T& max(const T& a, const T& b) {
  return (a<b)?b:a;
}

template <class T> const T& min(const T& a, const T& b) {
  return (a<b)?a:b;
}
