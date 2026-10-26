//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Utility Functions
//  **********************************

// Suppress Armadillo warnings (e.g., singular matrix warnings from solve())
// Must be defined BEFORE including Armadillo headers
#ifndef ARMA_DONT_PRINT_ERRORS
#define ARMA_DONT_PRINT_ERRORS
#endif

// my header file
# include <RcppArmadillo.h>
# include <xoshiro.h>
# include <dqrng_distribution.h>
# include <limits>
# include <cmath>
# include <stdexcept>

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
 size_t nsplit = 1;
 bool replacement = 0;
 double resample_prob = 0.8;
 bool useobsweight = 0;
 bool usevarprob = 0;
 size_t importance = 0;
 bool reinforcement = 0;
 
 // other control parameters  
 bool obs_track = 0;
 size_t var_mode = 0; // 0=none, 1=matched, 2=IJ, 3=jack
 size_t linear_comb = 1;
 double alpha = 0;
 size_t split_rule = 1;
 size_t linear_comb_method = 1; // LC method: naive(1), sir(2), pca(3), coxph(1), etc. separate from split_rule
 
// RLT parameters 
size_t embed_ntrees = 0;
double embed_mtry = 0;
size_t embed_nmin = 0;  
size_t embed_nsplit = 0;
 bool embed_replacement = 0;
 double embed_resample_prob = 0;
 double embed_mute = 0;
 size_t embed_protect = 0;
 double embed_threshold = 0.25;
 
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
  nsplit        = param["nsplit"];
   replacement   = param["resample.replace"];
   resample_prob = param["resample.prob"];
   useobsweight  = param["use.obs.w"];
   usevarprob  = param["use.var.prob"];    
   importance    = param["importance"];    
   reinforcement = param["reinforcement"];
   
   // other control parameters
   obs_track     = param["resample.track"];
   var_mode      = param["var.mode"];
   linear_comb   = param["linear.comb"];
   alpha         = param["alpha"];
   
  // Read split rule: linear.comb.method for regression, split.rule for class/surv
  // R-side may pass either integer or numeric, so use double conversion
  
 if (param.containsElementNamed("linear.comb.method"))
 {
   SEXP sr = param["linear.comb.method"];
   if (Rf_length(sr) > 0)
     linear_comb_method = (size_t) Rcpp::as<double>(sr);
   else
     linear_comb_method = 1;
 }
 
 if (param.containsElementNamed("split.rule"))
 {
   SEXP sr = param["split.rule"];
   if (Rf_length(sr) > 0)
     split_rule = (size_t) Rcpp::as<double>(sr);
   else
     split_rule = 1;
 }
   
   
   
  // RLT parameters
  embed_ntrees           = param["embed.ntrees"];
  embed_mtry             = param["embed.mtry"];
  embed_nmin             = param["embed.nmin"];  
  embed_nsplit           = param["embed.nsplit"];
   embed_replacement      = param["embed.resample.replace"];
   embed_resample_prob    = param["embed.resample.prob"];
   embed_mute             = param["embed.mute"];
   embed_protect          = param["embed.protect"];
   embed_threshold        = param["embed.threshold"];
   
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
  nsplit        = Input.nsplit;
   replacement   = Input.replacement;
   resample_prob = Input.resample_prob;
   useobsweight  = Input.useobsweight;
   usevarprob  = Input.usevarprob;
   importance    = Input.importance;  
   reinforcement = Input.reinforcement;
   
   // other control parameters   
   obs_track     = Input.obs_track;      
   var_mode      = Input.var_mode;
   linear_comb   = Input.linear_comb;
   alpha         = Input.alpha;
   split_rule    = Input.split_rule;
   linear_comb_method = Input.linear_comb_method;
   
  // RLT parameters 
  embed_ntrees           = Input.embed_ntrees;
  embed_mtry             = Input.embed_mtry;
  embed_nmin             = Input.embed_nmin;
  embed_nsplit           = Input.embed_nsplit;
   embed_replacement      = Input.embed_replacement;
   embed_resample_prob    = Input.embed_resample_prob;
   embed_mute             = Input.embed_mute;
   embed_protect          = Input.embed_protect;
   embed_threshold        = Input.embed_threshold;
   
   // system related
   ncores        = Input.ncores;
   verbose       = Input.verbose;
   seed          = Input.seed;
 };
 
 void print() const {
   
   RLTcout << "---------- Parameters Summary ----------" << std::endl;
   RLTcout << "              (N, P) = (" << N << ", " << P << ")" << std::endl;
   RLTcout << "          # of trees = " << ntrees << std::endl;
   RLTcout << "        (mtry, nmin) = (" << mtry << ", " << nmin << ")" << std::endl;
   
  if (nsplit == 0)
    RLTcout << "      split generate = Best" << std::endl;
  else
    RLTcout << "      split generate = Random, " << nsplit << std::endl;
   
   RLTcout << "            sampling = " << resample_prob << (replacement ? " w/ replace" : " w/o replace") << std::endl;
   
   RLTcout << "  (Obs, Var) weights = (" << (useobsweight ? "Yes" : "No") << ", " << (usevarprob ? "Yes" : "No") << ")" << std::endl;
   
   if (alpha > 0)
     RLTcout << "               alpha = " << alpha << std::endl;
   
   if (linear_comb > 1)
     RLTcout << "  linear combination = " << linear_comb << std::endl;
   
   if (linear_comb > 1)
     RLTcout << "      LC method code = " << linear_comb_method << std::endl;
   
   if (split_rule > 1)
     RLTcout << "   survival split ru = " << split_rule << std::endl;
   
   RLTcout << "          importance = " << (importance == 2 ? "distribute" : (importance == 1 ? "permute" : "none")) << std::endl;
   RLTcout << "       reinforcement = " << (reinforcement ? "Yes" : "No") << std::endl;
   RLTcout << "              ncores = " << ncores << std::endl;
   RLTcout << "                seed = " << seed << std::endl;
   RLTcout << "----------------------------------------" << std::endl;
   if (reinforcement) rlt_print();
 };
 
 void rlt_print() const {
   
   RLTcout << " embed.ntrees            = " << embed_ntrees << std::endl;

   if (embed_mtry < 1)
     RLTcout << " embed.mtry              = " << std::setprecision(3) << embed_mtry * 100 << "%" << std::endl;
   
   if (embed_mtry >= 1)
     RLTcout << " embed.mtry              = " << embed_mtry << std::endl;
   
   RLTcout << " embed.nmin              = " << embed_nmin << std::endl;
   
  if (embed_nsplit == 0)
    RLTcout << " embed.split.gen         = Best" << std::endl;
  else
    RLTcout << " embed.split.gen         = Random, " << embed_nsplit << std::endl;
   
   RLTcout << " embed.resample.replace  = " << (embed_replacement ? "TRUE" : "FALSE") << std::endl;
   RLTcout << " embed.resample_prob     = " << embed_resample_prob << std::endl;
   RLTcout << " embed.mute              = " << embed_mute << std::endl;
   RLTcout << " embed.protect           = " << embed_protect << std::endl;
   RLTcout << " embed.threshold         = " << embed_threshold << std::endl;
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

void cumsum_rev(arma::uvec& seq);
void cumsum_rev(arma::vec& seq);

// *************** //
// field functions //
// *************** //

void field_vec_resize(arma::field<arma::vec>& A, size_t size);
void field_vec_resize(arma::field<arma::uvec>& A, size_t size);
void field_vec_resize(arma::field<arma::ivec>& A, size_t size);

// ************************//
// Random Number Generator //
// ************************//

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
  // generate number [min, max]
  size_t rand_sizet(size_t min, size_t max){
    
    boost::random::uniform_int_distribution<size_t> rand(min, max);
    
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
    
    boost::random::uniform_int_distribution<size_t> rand(min, max);
    
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
      Num = N;
    }

    //boost::uniform_01<dqrng::xoshiro256plus> rand(this -> lrng);
      
    //boost::random::uniform_real_distribution<double> rand(0, 1);
      
    for (size_t i = 0; i < Num; i++){

      boost::random::uniform_int_distribution<size_t> rand(i, N-1);
      
      size_t randomloc = rand(this->lrng);
      
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
      
      boost::random::uniform_int_distribution<size_t> rand(min, max);
      
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
  
  // Weighted sampling without replacement (Efraimidis-Spirakis)
  // x: vector of elements to sample from
  // Num: number of elements to draw
  // weight: probability weights (same length as x)
  // Returns a subset of x of size min(Num, x.n_elem), drawn without replacement
  // proportional to weight.
  template<typename T> T weighted_sample(T x, const arma::vec& weight, size_t Num) {
    
    size_t N = x.n_elem;
    
    if (Num > N) 
      Num = N;
    
    if (Num == N)
      return x;

    if (arma::accu(weight > 0) < Num)
      throw std::invalid_argument("Variable sampling has fewer positive weights than requested candidates.");
    
    // working copies
    arma::vec w = weight;            // mutable weights
    arma::uvec indices(N);           // active index map
    size_t n_active = N;
    double total_weight = arma::sum(w);
    
    for (size_t i = 0; i < N; i++)
      indices(i) = i;
    
    arma::uvec result(Num);
    
    for (size_t k = 0; k < Num; k++)
    {
      if (!(total_weight > 0) || !std::isfinite(total_weight))
        throw std::invalid_argument("Variable sampling has no finite positive remaining weight.");

      // draw U ~ [0, total_weight)
      boost::random::uniform_real_distribution<double> rand(0.0, total_weight);
      double u = rand(this->lrng);
      
      // locate element via cumulative sum
      double cumsum = 0.0;
      size_t idx = 0;
      for (idx = 0; idx < n_active - 1; idx++)
      {
        cumsum += w(idx);
        if (cumsum > u) break;
      }
      
      // selected element
      result(k) = x(indices(idx));
      
      // remove selected: swap to end, shrink
      total_weight -= w(idx);
      w(idx) = w(n_active - 1);
      indices(idx) = indices(n_active - 1);
      n_active--;
    }
    
    return result;
  }
  
};

#endif
