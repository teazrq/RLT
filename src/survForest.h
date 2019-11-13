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

List SurvForestUniFit(arma::mat& X,
                      arma::uvec& Y,
                      arma::uvec& Censor,
                      arma::uvec& Ncat,
                      List& param,
                      List& RLTparam,
                      arma::vec& obsweight,
                      arma::vec& varweight,
                      int usecores,
                      int verbose,
                      arma::umat& ObsTrackPre);

void Surv_Uni_Forest_Build(const RLT_SURV_DATA& REG_DATA,
                           Surv_Uni_Forest_Class& SURV_FOREST,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& Param_RLT,
                           uvec& obs_id,
                           uvec& var_id,
                           umat& ObsTrack,
                           mat& Prediction,
                           mat& OOBPrediction,
                           arma::field<arma::field<arma::uvec>>& NodeRegi,
                           vec& VarImp,
                           size_t seed,
                           int usecores,
                           int verbose);

void Surv_Uni_Split_A_Node(size_t Node,
                           Surv_Uni_Tree_Class& OneTree,
                           arma::field<arma::uvec>& OneNodeRegi,
                           const RLT_SURV_DATA& SURV_DATA,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& Param_RLT,
                           uvec& obs_id,
                           uvec& var_id);

void Surv_Uni_Terminate_Node(size_t Node, 
                             Surv_Uni_Tree_Class& OneTree,
                             arma::field<arma::uvec>& OneNodeRegi,
                             uvec& obs_id,
                             const uvec& Y,
                             const uvec& Censor,
                             const size_t NFail,
                             const vec& obs_weight,
                             const PARAM_GLOBAL& Param,
                             bool useobsweight);

void Surv_Uni_Find_A_Split(Uni_Split_Class& OneSplit,
                           const RLT_SURV_DATA& SURV_DATA,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& RLTParam,
                           uvec& obs_id,
                           uvec& var_id);

void Surv_Uni_Split_Cont(Uni_Split_Class& TempSplit, 
                         uvec& obs_id,
                         const vec& x,
                         const uvec& Y, // Y is collapsed
                         const uvec& Censor, // Censor is collapsed
                         vec& obs_weight,
                         size_t NFail,
                         double penalty,
                         int split_gen,
                         int split_rule,
                         int nsplit,
                         size_t nmin, 
                         double alpha,
                         bool useobsweight,
                         bool failforce);
    
void Surv_Uni_Split_Cat(Uni_Split_Class& TempSplit, 
                         uvec& obs_id,
                         const vec& x,
                         const uvec& Y, // Y is collapsed
                         const uvec& Censor, // Censor is collapsed
                         vec& obs_weight,
                         size_t NFail,
                         double penalty,
                         int split_gen,
                         int split_rule,
                         int nsplit,
                         size_t nmin, 
                         double alpha,
                         bool useobsweight,
                         bool failforce,
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
                              const uvec& Y, // this is collapsed 
                              const uvec& Censor, // this is collapsed 
                              const vec& obs_weight,
                              size_t NFail,
                              double a_random_cut,
                              int split_rule,
                              double penalty,
                              bool useobsweight);

double surv_cont_score_at_index(uvec& obs_ranked, // collapsed
                                uvec& indices, // this is not collapsed, indicates original id
                                const uvec& Y, //collapsed
                                const uvec& Censor, //collapsed
                                const vec& obs_weight,
                                size_t NFail,
                                size_t a_random_ind,
                                int split_rule,
                                double penalty, 
                                bool useobsweight);

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

double logrank(vec& Left_Count_Fail,
               vec& Left_Count_Censor,
               vec& Right_Count_Fail,
               vec& Right_Count_Censor,
               double LeftN,
               double N,
               size_t NFail);

double suplogrank(vec& Left_Count_Fail,
                  vec& Left_Count_Censor,
                  vec& Right_Count_Fail,
                  vec& Right_Count_Censor,
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

vec hazard(const vec& Fail, const vec& Censor);

double survloglike(const vec& basehazard, const vec& lefthazard, const vec& righthazard, 
                   const vec& Left_Count_Fail, const vec& Left_Count_Censor, 
                   const vec& Right_Count_Fail, const vec& Right_Count_Censor, 
                   double penalty);

// utility functions 


// prediction 

void Surv_Uni_Forest_Pred(cube& Pred,
                          mat& W,
                          const Surv_Uni_Forest_Class& SURV_FOREST,
                          const mat& X,
                          const uvec& Ncat,
                          size_t NFail,
                          bool kernel,
                          int usecores,
                          int verbose);

// for converting 

List surv_uni_convert_forest_to_r(std::vector<Surv_Uni_Tree_Class>& Forest);
    
#endif
