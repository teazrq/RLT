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
                        const uvec& Y, // Y is collapsed
                        const uvec& Censor, // Censor is collapsed
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        size_t nmin,
                        double alpha,
                        vec& obs_weight,
                        bool useobsweight,
                        size_t NFail,
                        int failforce,
                        size_t ncat);

void collapse(const uvec& Y, 
              const uvec& Censor, 
              uvec& Y_collapse, 
              uvec& Censor_collapse, 
              uvec& obs_id,
              size_t& NFail);

// splitting score calculations

double surv_cont_score_at_cut(uvec& obs_id,
                              const vec& x,
                              const uvec& Y,
                              const uvec& Censor,
                              size_t NFail,
                              double a_random_cut,
                              int split_rule);

double surv_cont_score_at_cut_w(uvec& obs_id,
                                const vec& x,
                                const uvec& Y,
                                const uvec& Censor,
                                size_t NFail,
                                double a_random_cut,
                                vec& obs_weight,
                                int split_rule);

double surv_cont_score_at_index(uvec& indices,
                                const uvec& Y, 
                                const uvec& Censor, 
                                size_t NFail,
                                size_t a_random_ind,
                                int split_rule);

double surv_cont_score_at_index_w(uvec& indices,
                                  const uvec& Y, 
                                  const uvec& Censor, 
                                  size_t NFail,
                                  size_t a_random_ind,
                                  vec& obs_weight,
                                  int split_rule);

double surv_cont_score_best(uvec& indices, // for x, sorted
                            const vec& x,
                            uvec& obs_ranked, // for Y and censor, sorted
                            const uvec& Y, 
                            const uvec& Censor, 
                            size_t NFail, 
                            size_t lowindex, 
                            size_t highindex, 
                            double& temp_cut, 
                            double& temp_score,
                            int split_rule);

double surv_cont_score_best_w(uvec& indices,
                              const vec& x,
                              uvec& obs_ranked,
                              const uvec& Y, 
                              const uvec& Censor, 
                              size_t NFail, 
                              size_t lowindex, 
                              size_t highindex, 
                              double& temp_cut, 
                              double& temp_score,
                              vec& obs_weight,
                              int split_rule);

double surv_cat_score(std::vector<Surv_Cat_Class>& cat_reduced, 
                      size_t temp_cat, 
                      size_t true_cat,
                      size_t NFail, 
                      int split_rule,
                      bool useobsweight);
    
void surv_cat_score_best(std::vector<Surv_Cat_Class>& cat_reduced,
                         size_t lowindex,
                         size_t highindex,
                         size_t true_cat,
                         size_t& temp_cat,
                         double& temp_score,
                         size_t NFail,
                         int split_rule,
                         bool useobsweight);

double logrank(uvec& Left_Count_Fail,
               uvec& Left_Count_Censor,
               uvec& Right_Count_Fail,
               uvec& Right_Count_Censor,
               double LeftN,
               double N,
               size_t NFail);

double suplogrank(uvec& Left_Count_Fail,
                  uvec& Left_Count_Censor,
                  uvec& Right_Count_Fail,
                  uvec& Right_Count_Censor,
                  double LeftN,
                  double N,
                  size_t NFail);

double logrank_w(vec& Left_Count_Fail,
                 vec& Left_Count_Censor,
                 vec& Right_Count_Fail,
                 vec& Right_Count_Censor,
                 double LeftW,
                 double W,
                 size_t NFail, 
                 bool useobsweight);

double suplogrank_w(vec& Left_Count_Fail,
                    vec& Left_Count_Censor,
                    vec& Right_Count_Fail,
                    vec& Right_Count_Censor,
                    double LeftW,
                    double W,
                    size_t NFail,
                    bool useobsweight);

// utility functions 


// prediction 

mat Surv_Uni_Forest_Pred(const std::vector<Surv_Uni_Tree_Class>& Forest,
                         const mat& X,
                         const uvec& Ncat,
                         int NFail,
                         bool kernel,
                         int usecores,
                         int verbose);

// for converting 

List surv_uni_convert_forest_to_r(std::vector<Surv_Uni_Tree_Class>& Forest);
    
#endif
