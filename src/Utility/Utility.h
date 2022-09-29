//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Utility Functions
//  **********************************

// my header file
# include <RcppArmadillo.h>
# include <Rcpp.h>
# include <xoshiro.h>
# include <dqrng_distribution.h>
# include <limits>

using namespace Rcpp;
using namespace arma;

// print function for R / python 

#ifndef RLT_PRINT
#define RLTcout Rcout
#endif

// ****************//
//  OMP functions  //
// ****************//

#ifdef _OPENMP
#include <omp.h>
#define OMPMSG(...)
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#define OMPMSG(...) RLTcout << "Package is not compiled with OpenMP (omp.h).\n" << std::endl;
#endif

// ******* //
//  Debug  //
// ******* //

// this debug function will output to R
//#ifdef RLT_DEBUG
//#define DEBUG_Rcout Rcout
//#else
//#define DEBUG_Rcout 0 && Rcout
//#endif

// this debug function will output results to a .txt file
/* void printLog(const char*, const char*, const int, const double);

#ifdef RLT_DEBUG
#define DEBUGPRINT(mode, x, n1, n2) printLog(mode, x, n1, n2)
#else
#define DEBUGPRINT(mode, x, n1, n2)
#endif */

#ifndef RLT_UTILITY
#define RLT_UTILITY

// ****************//
// Check functions //
// ****************//

size_t checkCores(size_t, size_t);

// *************//
// Calculations //
// *************//

template <class T> const T& max (const T& a, const T& b);
template <class T> const T& min (const T& a, const T& b);

// ************************//
// Random Number Generator //
// ************************//

// int intRand(const int & min, const int & max);

// Structure for Random Number generating
class Rand{
  
public:
  
  size_t seed = 0;
  dqrng::xoshiro256plus lrng; // Random Number Generator
  
  // Initialize
  Rand(size_t seed){
    dqrng::xoshiro256plus rng(seed);
    lrng = rng;
  }
  
  // Random
  size_t rand_sizet(size_t min, size_t max){
    
    boost::random::uniform_int_distribution<int> rand(min, max);
    
    return  rand(this -> lrng);
    
  };
  
  // Random 01
  double rand_01(){
    
    //boost::uniform_01<dqrng::xoshiro256plus> rand(this -> lrng);
    boost::random::uniform_real_distribution<double> rand(0, 1);
    return  rand(this -> lrng);
  };
  
  // Discrete Uniform
  arma::uvec rand_uvec(size_t min, size_t max, size_t Num){
    
    if (max < min) max = min;
    
    boost::random::uniform_int_distribution<int> rand(min, max);
    
    arma::uvec x(Num);
    
    for(size_t i = 0; i < Num; i++){
      
      x(i) = rand(this -> lrng);
      
    }
    
    return x;
    
  };
  
  // Uniform Distribution
  arma::vec rand_vec(double min, double max, size_t Num){

    if (max < min) max = min;
    
    boost::random::uniform_real_distribution<double> rand(min, max);
    
    arma::vec x(Num);
    
    for(size_t i = 0; i < Num; i++){
      
      x(i) = rand(this -> lrng);
      
    }
    
    return x;
    
  };
  
  // Sampling in a range without replacement
  arma::uvec sample(size_t min, size_t max, size_t Num) {

    if (max < min) max = min;

    size_t N = max - min + 1;

    arma::uvec x = arma::linspace<uvec>(min, max, N);
    
    if (Num > N) 
    {
      RLTcout << "Num = " << Num << " N = " << N << " min = " << min << " max = " << max << std::endl;
      Num = N;
    }

    //boost::uniform_01<dqrng::xoshiro256plus> rand(this -> lrng);
      
    //boost::random::uniform_real_distribution<double> rand(0, 1);
      
    for (size_t i = 0; i < Num; i++){

      boost::random::uniform_int_distribution<int> rand(i, N-1);
      
      size_t randomloc = rand(this->lrng);
      
      //size_t randomloc = i + (size_t) (N-i)*rand(this -> lrng);
      
      //size_t randomloc = i + (size_t) (N-i)*rand();
      
      // swap
      size_t temp = x(i);
      x(i) = x(randomloc);
      x(randomloc) = temp;
      
    }
    
    x.resize(Num);
    
    return x;
    
  };
  
  arma::uvec sample(size_t min, size_t max, size_t Num, bool replace) {
    
    if (replace == 0)
      return this->sample(min, max, Num);
    else{
      
      if (max < min) max = min;
      
      //size_t N = max - min + 1;
      
      //boost::uniform_01<dqrng::xoshiro256plus> rand(this -> lrng);
      
      boost::random::uniform_int_distribution<int> rand(min, max);
      
      arma::uvec x(Num);
      
      for(size_t i = 0; i < Num; i++){
        
        //x(i) = min + (size_t) N*rand();
        x(i) = rand(this->lrng);
      }
      
      return x;
      
    }
    
  };

  // Sampling a vector without replacement
  template<typename T> T sample(T x, size_t Num) {
    
    size_t N = x.n_elem;
    
    arma::uvec loc = this->sample(0, N-1, Num);
    
    return x(loc);
    
  }
  
  // shuffle
  template<typename T> T shuffle(T z){
    
    arma::uvec temp = this->sample(0, z.n_elem -1, z.n_elem);
    
    T z_shuffle = z(temp);
    
    return z_shuffle;
  }
  
};

#endif