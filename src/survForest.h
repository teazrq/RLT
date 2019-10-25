//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility//Utility.h"

using namespace Rcpp;
using namespace arma;

#ifndef RegForest_Fun
#define RegForest_Fun

// univariate tree split functions 

List SurvForestUniFit(DataFrame& X,
          					 uvec& Y,
          					 uvec& Censor,
          					 uvec& Ncat,
          					 List& param,
          					 List& RLTparam,
          					 vec& obsweight,
          					 vec& varweight,
          					 int usecores,
          					 int verbose);

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
            						  cube& Pred,
            						  arma::field<arma::field<arma::uvec>>& NodeRegi,
            						  vec& VarImp,
            						  int seed,
            						  int usecores,
            						  int verbose);

void Surv_Uni_Split_A_Node(size_t Node,
                           Surv_Uni_Tree_Class& OneTree,
                           std::vector<arma::uvec>& OneNodeRegi,
                           const mat& X,
                           const uvec& Y,
                           const uvec& Censor,
                           const size_t NFail, 
                           const uvec& Ncat,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& Param_RLT,
                           vec& obs_weight,
                           uvec& obs_id,
                           vec& var_weight,
                           uvec& var_id);

void Surv_Uni_Terminate_Node(size_t Node, 
                             Surv_Uni_Tree_Class& OneTree,
                             std::vector<arma::uvec>& OneNodeRegi,
                             const uvec& Y,
                             const uvec& Censor,
                             const size_t NFail,
                             const PARAM_GLOBAL& Param,
                             vec& obs_weight,
                             uvec& obs_id,
                             bool useobsweight);

void Surv_Uni_Find_A_Split(Uni_Split_Class& OneSplit,
                           const mat& X,
                           const uvec& Y,
                           const uvec& Censor,
                           const uvec& Ncat,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& RLTParam,
                           vec& obs_weight,
                           uvec& obs_id,
                           vec& var_weight,
                           uvec& var_id);

void Surv_Uni_Split_Cont(Uni_Split_Class& TempSplit, 
                         uvec& obs_id,
                         const vec& x,
                         const uvec& Y,
                         const uvec& Censor,
                         double penalty,
                         int split_gen,
                         int split_rule,
                         int nsplit,
                         size_t nmin, 
                         double alpha,
                         vec& obs_weight,
                         bool useobsweight,
                         size_t nfail,
                         int failforce);
    
void Surv_Uni_Split_Cat(Uni_Split_Class& TempSplit,
                        uvec& obs_id,
                        const vec& x,
                        const uvec& Y,
                        const uvec& Censor,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        size_t nmin,
                        double alpha,
                        vec& obs_weight,
                        bool useobsweight,
                        size_t ncat);

void collapse(const uvec& Y, 
              const uvec& Censor, 
              uvec& Y_collapse, 
              uvec& Censor_collapse, 
              uvec& obs_id,
              size_t& NFail);

// splitting score calculations

double logrank(uvec& Left_Count_Fail,
               uvec& Left_Count_Censor,
               uvec& Right_Count_Fail,
               uvec& Right_Count_Censor,
               size_t LeftN,
               size_t N,
               size_t nfail);


    
#endif
