//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Utility Functions
//  **********************************

// my header file
# include <RcppArmadillo.h>
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
// Get Parameters  //
// ****************//

class PARAM_GLOBAL{
public:
 
 // main parameters
 size_t N = 0;
 size_t P = 0;
 size_t ntrees = 1;
 size_t mtry = 1;
 size_t nmin = 1;
 size_t split_gen = 1;
 size_t nsplit = 1;
 bool replacement = 0;
 double resample_prob = 0.8;
 bool useobsweight = 0;
 bool usevarweight = 0;
 bool importance = 0;
 bool reinforcement = 0;
 
 // other control parameters  
 bool obs_track = 0;
 size_t linear_comb = 1;
 double alpha = 0;
 size_t split_rule = 1;
 //size_t varweighttype = 0;  
 bool failcount = 0;  
 
 // RLT parameters 
 size_t embed_ntrees = 0;
 double embed_mtry = 0;
 size_t embed_nmin = 0;  
 size_t embed_split_gen = 0;
 size_t embed_nsplit = 0;  
 double embed_resample_prob = 0;
 double embed_mute = 0;
 size_t embed_protect = 0;  
 
 // system related
 size_t ncores = 1;
 size_t verbose = 0;
 size_t seed = 1;
 
 void PARAM_READ_R(List& param){
   
   // main parameters
   N             = param["n"];
   P             = param["p"];
   ntrees        = param["ntrees"];
   mtry          = param["mtry"];
   nmin          = param["nmin"];
   split_gen     = param["split.gen"];
   nsplit        = param["nsplit"];
   replacement   = param["resample.replace"];
   resample_prob = param["resample.prob"];
   useobsweight  = param["use.obs.w"];
   usevarweight  = param["use.var.w"];    
   importance    = param["importance"];    
   reinforcement = param["reinforcement"];
   
   // other control parameters
   obs_track     = param["resample.track"];
   linear_comb   = param["linear.comb"];
   alpha         = param["alpha"];
   split_rule    = param["split.rule"];
   //varweighttype = param["var.w.type"];
   failcount     = param["failcount"];
   
   // RLT parameters
   embed_ntrees        = param["embed.ntrees"];
   embed_mtry          = param["embed.mtry"];
   embed_nmin          = param["embed.nmin"];  
   embed_split_gen     = param["embed.split.gen"];
   embed_nsplit        = param["embed.nsplit"];    
   embed_resample_prob = param["embed.resample.prob"];
   embed_mute          = param["embed.mute"];
   embed_protect       = param["embed.protect"];  
   
   // system related
   ncores        = param["ncores"];
   verbose       = param["verbose"];
   seed          = param["seed"];
 };
 
 void copyfrom(const PARAM_GLOBAL& Input){
   // main parameters
   N             = Input.N;
   P             = Input.P;
   ntrees        = Input.ntrees;
   mtry          = Input.mtry;
   nmin          = Input.nmin;
   split_gen     = Input.split_gen;
   nsplit        = Input.nsplit;
   replacement   = Input.replacement;
   resample_prob = Input.resample_prob;
   useobsweight  = Input.useobsweight;
   usevarweight  = Input.usevarweight;
   importance    = Input.importance;  
   reinforcement = Input.reinforcement;
   
   // other control parameters   
   obs_track     = Input.obs_track;      
   linear_comb   = Input.linear_comb;
   alpha         = Input.alpha;
   split_rule    = Input.split_rule;
   //varweighttype = Input.varweighttype;  
   failcount     = Input.failcount;
   
   // RLT parameters 
   embed_ntrees        = Input.embed_ntrees;
   embed_mtry          = Input.embed_mtry;
   embed_nmin          = Input.embed_nmin;
   embed_split_gen     = Input.embed_split_gen;
   embed_nsplit        = Input.embed_nsplit;
   embed_resample_prob = Input.embed_resample_prob;
   embed_mute          = Input.embed_mute;
   embed_protect       = Input.embed_protect;  
   
   // system related
   ncores        = Input.ncores;
   verbose       = Input.verbose;
   seed          = Input.seed;
 };
 
 void print() {
   
   RLTcout << "---------- Parameters Summary ----------" << std::endl;
   RLTcout << "              (N, P) = (" << N << ", " << P << ")" << std::endl;
   RLTcout << "          # of trees = " << ntrees << std::endl;
   RLTcout << "        (mtry, nmin) = (" << mtry << ", " << nmin << ")" << std::endl;
   
   if (split_gen == 3)
     RLTcout << "      splitting rule = Best" << std::endl;
   
   if (split_gen < 3)
     RLTcout << "      splitting rule = " << ((split_gen == 1) ? "Random, " : "Rank, ") << nsplit << std::endl;
   
   RLTcout << "            sampling = " << resample_prob << (replacement ? " w/ replace" : " w/o replace") << std::endl;
   
   RLTcout << "  (Obs, Var) weights = (" << (useobsweight ? "Yes" : "No") << ", " << (usevarweight ? "Yes" : "No") << ")" << std::endl;
   
   if (alpha > 0)
     RLTcout << "               alpha = " << alpha << std::endl;
   
   if (linear_comb > 1)
     RLTcout << "  linear combination = " << linear_comb << std::endl;
   
   RLTcout << "       reinforcement = " << (reinforcement ? "Yes" : "No") << std::endl;
   RLTcout << "----------------------------------------" << std::endl;
   if (reinforcement) rlt_print();
 };
 
 void print() const {
   
   RLTcout << "---------- Parameters Summary ----------" << std::endl;
   RLTcout << "              (N, P) = (" << N << ", " << P << ")" << std::endl;
   RLTcout << "          # of trees = " << ntrees << std::endl;
   RLTcout << "        (mtry, nmin) = (" << mtry << ", " << nmin << ")" << std::endl;
   
   if (split_gen == 3)
     RLTcout << "      splitting rule = Best" << std::endl;
   
   if (split_gen < 3)
     RLTcout << "      splitting rule = " << ((split_gen == 1) ? "Random, " : "Rank, ") << nsplit << std::endl;
   
   RLTcout << "            sampling = " << resample_prob << (replacement ? " w/ replace" : " w/o replace") << std::endl;
   
   RLTcout << "  (Obs, Var) weights = (" << (useobsweight ? "Yes" : "No") << ", " << (usevarweight ? "Yes" : "No") << ")" << std::endl;
   
   if (alpha > 0)
     RLTcout << "               alpha = " << alpha << std::endl;
   
   if (linear_comb > 1)
     RLTcout << "  linear combination = " << linear_comb << std::endl;
   
   RLTcout << "       reinforcement = " << (reinforcement ? "Yes" : "No") << std::endl;
   RLTcout << "----------------------------------------" << std::endl;
   if (reinforcement) rlt_print();
 };
 
 void rlt_print() {
   
   RLTcout << " embed.ntrees        = " << embed_ntrees << std::endl;
   RLTcout << " embed.mtry          = " << embed_mtry << std::endl;    
   RLTcout << " embed.nmin          = " << embed_nmin << std::endl;
   RLTcout << " embed.split_gen     = " << embed_split_gen << std::endl;
   RLTcout << " embed.nsplit        = " << embed_nsplit << std::endl;    
   RLTcout << " embed.resample_prob = " << embed_resample_prob << std::endl;
   RLTcout << " embed.mute          = " << embed_mute << std::endl;
   RLTcout << " embed.protect       = " << embed_protect << std::endl;
   RLTcout << "----------------------------------------" << std::endl;
   
 };
 
 void rlt_print() const {
   
   RLTcout << " embed.ntrees        = " << embed_ntrees << std::endl;
   RLTcout << " embed.mtry          = " << embed_mtry << std::endl;    
   RLTcout << " embed.nmin          = " << embed_nmin << std::endl;
   RLTcout << " embed.split_gen     = " << embed_split_gen << std::endl;
   RLTcout << " embed.nsplit        = " << embed_nsplit << std::endl;    
   RLTcout << " embed.resample_prob = " << embed_resample_prob << std::endl;
   RLTcout << " embed.mute          = " << embed_mute << std::endl;
   RLTcout << " embed.protect       = " << embed_protect << std::endl;
   RLTcout << "----------------------------------------" << std::endl;
   
 };
}; 
 
// ****************//
// Check functions //
// ****************//

size_t checkCores(size_t, size_t);

// *************//
// Calculations //
// *************//

template <class T> const T& max (const T& a, const T& b);
template <class T> const T& min (const T& a, const T& b);

void cumsum_rev(arma::uvec& seq);

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