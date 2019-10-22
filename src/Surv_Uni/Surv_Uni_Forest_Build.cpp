//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"

#include <xoshiro.h>
#include <dqrng_distribution.h>

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Forest_Build(const mat& X,
            						   const uvec& Y,
            						   const uvec& Censor,
            						   const uvec& Ncat,
            						   const PARAM_GLOBAL& Param,
            						   const PARAM_RLT& Param_RLT,
            						   vec& obs_weight,
            						   uvec& obs_id,
            						   vec& var_weight,
            						   uvec& var_id,
            						   std::vector<Surv_Uni_Tree_Class>& Forest,
            						   imat& ObsTrack,
            						   mat& Pred,
            						   arma::field<arma::field<arma::uvec>>& NodeRegi,
            						   vec& VarImp,
            						   int seed,
            						   int usecores,
            						   int verbose)
{
  // parameters need to be used
  size_t ntrees = Param.ntrees;
  bool replacement = Param.replacement;
  double resample_prob = Param.resample_prob;
  size_t P = var_id.n_elem;
  size_t N = obs_id.n_elem;
  size_t size = (size_t) obs_id.n_elem*resample_prob;
  size_t nmin = Param.nmin;
  bool kernel_ready = Param.kernel_ready;

  // start parallel trees

  Rcout << std::endl << " --- survForestBuild " << std::endl;

  dqrng::xoshiro256plus rng(seed); // properly seeded rng
    
  #pragma omp parallel num_threads(usecores)
  {
    
    dqrng::xoshiro256plus lrng(rng);      // make thread local copy of rng 
    lrng.long_jump(omp_get_thread_num() + 1);  // advance rng by 1 ... ncores jumps
    
  #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++) // fit all trees
    {
      
      DEBUG_Rcout << "-- Fitting tree " << nt << std::endl;
      
      // get inbag and oobag samples
      uvec inbagObs, oobagObs;
      oob_samples(inbagObs, oobagObs, obs_id, size, replacement);
      
      // record to the ObsTrack matrix
      for (size_t i = 0; i < size; i++)
        ObsTrack(inbagObs(i), nt)++;
      
      DEBUG_Rcout << "-- Initiate tree " << nt << std::endl;
      
      // initialize a tree (univariate split)
      size_t TreeLength = 1 + size/nmin*3;
      Forest[nt].initiate(TreeLength, P);
      
      // define a temporary object to save node regi since field cannot be resized 
      std::vector<uvec> OneNdeRegi;
      
      if (kernel_ready)
        OneNdeRegi.resize(TreeLength);
      
      // start to fit a tree
      Forest[nt].NodeType(0) = 1; // 0: unused, 1: reserved; 2: internal node; 3: terminal node
      
      DEBUG_Rcout << "-- Build tree " << nt << std::endl;
      
      Surv_Uni_Split_A_Node(0, Forest[nt], OneNdeRegi,
                            X, Y, Censor, Ncat, Param, Param_RLT,
                            obs_weight, inbagObs, var_weight, var_id);
      
      DEBUG_Rcout << "-- Record tree " << nt << std::endl;
      
      // trim and record tree
      // DEBUG_Rcout << "-- Forest[nt].NodeType " << Forest[nt].NodeType << std::endl;
      
      // DEBUG_Rcout << "-- Forest[nt].NodeSurv " << Forest[nt].NodeSurv << std::endl;
      
      TreeLength = Forest[nt].get_tree_length();
      
      DEBUG_Rcout << "-- This tree length is " << TreeLength << std::endl;
      
      Forest[nt].trim(TreeLength);
      
      DEBUG_Rcout << "-- Forest[nt].NodeSurv " << Forest[nt].NodeSurv << std::endl;
      
      DEBUG_Rcout << "-- Record noderegi " << nt << std::endl;
      
      // record noderegi
      if (kernel_ready)
      {
        NodeRegi[nt].set_size(TreeLength);
        
        for (size_t i=0; i < TreeLength; i++)
          if (Forest[nt].NodeType(i) == 3)
            NodeRegi[nt][i] = uvec(&(OneNdeRegi[i][0]), OneNdeRegi[i].n_elem, false, true);
      }
      
      DEBUG_Rcout << "-- calcualte oob prediciton " << std::endl;
      
      // predictions 
      
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
      
      Uni_Find_Terminal_Node(0, Forest[nt], X, Ncat, proxy_id, obs_id, TermNode);
      
      
      // Pred.col(nt) = Forest[nt].NodeAve(TermNode);
      
    }
    
  #pragma omp barrier
    
  #pragma omp for schedule(static)
    for (size_t i = 0; i < N; i++) // fit all trees
    {
      // Rcout << "-- finish up prediction for subject " << i << std::endl;
    }
    
  }

}