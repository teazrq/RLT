//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival Functions
//  **********************************

// my header files
# include "../Utility/Tree_Definition.h"
# include "../Utility/Utility.h"
# include "../Utility/Tree_Function.h"
# include "Surv_Uni_Definition.h"

using namespace Rcpp;
using namespace arma;

#ifndef RLT_SURV_UNI_FUNCTION
#define RLT_SURV_UNI_FUNCTION

// univariate tree split functions 

List SurvUniForestFit(mat& X,
          	         uvec& Y,
          	         uvec& Censor,
          	         uvec& Ncat,
          		       vec& obsweight,
          		       vec& varweight,
          		       umat& ObsTrackPre,
          		       List& param);

void Surv_Uni_Forest_Build(const RLT_SURV_DATA& SURV_DATA,
                          Surv_Uni_Forest_Class& SURV_FOREST,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          umat& ObsTrack,
                          bool do_prediction,
                          mat& Prediction,
                          mat& OOBPrediction,
                          vec& VarImp);

void Surv_Uni_Split_A_Node(size_t Node,
                          Surv_Uni_Tree_Class& OneTree,
                          const RLT_SURV_DATA& SURV_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl);

void Surv_Uni_Split_A_Node_Embed(size_t Node,
                                Surv_Uni_Tree_Class& OneTree,
                                const RLT_SURV_DATA& SURV_DATA,
                                const PARAM_GLOBAL& Param,
                                uvec& obs_id,
                                const uvec& var_id,
                                const uvec& var_protect,
                                Rand& rngl);
  
void Surv_Uni_Terminate_Node(size_t Node, 
                            Surv_Uni_Tree_Class& OneTree,
                            uvec& obs_id,
                            const uvec& Y,
                            const uvec& Censor,
                            const size_t NFail,
                            const vec& obs_weight,
                            bool useobsweight);


void Surv_Uni_Find_A_Split(Split_Class& OneSplit,
                          const RLT_SURV_DATA& SURV_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl);

void Surv_Uni_Find_A_Split_Embed(Split_Class& OneSplit,
                                const RLT_SURV_DATA& SURV_DATA,
                                const PARAM_GLOBAL& Param,
                                const uvec& obs_id,
                                uvec& var_id,
                                uvec& var_protect,
                                Rand& rngl);

void Surv_Uni_Split_Cont(Split_Class& TempSplit,
                        const uvec& obs_id,
                        const vec& x,
                        const uvec& Y,
                        const uvec& Censor,
                        const size_t NFail,
                        uvec& All_Fail,
                        vec& All_Risk,
                        const vec& obs_weight,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl);

void Surv_Uni_Split_Cat(Split_Class& TempSplit,
                       const uvec& obs_id,
                       const vec& x,
                       const size_t ncat,
                       const uvec& Y,
                       const uvec& Censor,
                       const vec& obs_weight,
                       double penalty,
                       int split_gen,
                       int split_rule,
                       int nsplit,
                       double alpha,
                       bool useobsweight,
                       Rand& rngl);

// splitting score calculations (continuous)

double surv_cont_score_at_cut(const uvec& obs_id,
                             const vec& x,
                             const uvec& Y,
                             const uvec& Censor,
                             const size_t NFail,
                             uvec& All_Fail,
                             vec& All_Risk,
                             double a_random_cut,
                             int split_rule);

double surv_cont_score_at_index(uvec& indices,
                                uvec& obs_ranked,
                                const uvec& Y,
                               const uvec& Censor,
                               const size_t NFail,
                               uvec& All_Fail,
                               vec& All_Risk,
                               size_t a_random_ind,
                               int split_rule);

void surv_cont_score_best(uvec& indices,
                          uvec& obs_ranked,
                          const vec& x,
                         const uvec& Y,
                         const uvec& Censor,
                         const size_t NFail,
                         uvec& All_Fail,
                         vec& All_Risk,
                         size_t lowindex, 
                         size_t highindex, 
                         double& temp_cut, 
                         double& temp_score,
                         int split_rule);

// splitting score calculations (categorical)

double surv_cat_score(std::vector<Surv_Cat_Class>& cat_reduced, 
                     size_t temp_cat, 
                     size_t true_cat);

double surv_cat_score_w(std::vector<Surv_Cat_Class>& cat_reduced, 
                       size_t temp_cat, 
                       size_t true_cat);

void surv_cat_score_best(std::vector<Surv_Cat_Class>& cat_reduced, 
                        size_t lowindex,
                        size_t highindex,
                        size_t true_cat,
                        size_t& best_cat,
                        double& best_score);

void surv_cat_score_best_w(std::vector<Surv_Cat_Class>& cat_reduced, 
                          size_t lowindex,
                          size_t highindex,
                          size_t true_cat,
                          size_t& best_cat,
                          double& best_score);

// for prediction 

void Surv_Uni_Forest_Pred(cube& Pred,
                         const Surv_Uni_Forest_Class& SURV_FOREST,
                         const mat& X,
                         const uvec& Ncat,
                         size_t& NFail,
                         size_t usecores,
                         size_t verbose);

// Survival specific functions

void collapse(const uvec& Y, const uvec& Censor, 
              uvec& Y_collapse, uvec& Censor_collapse, 
              uvec& obs_id, size_t& NFail);

// splitting score calculations

double logrank(const uvec& Left_Fail, 
               const uvec& Left_Risk, 
               uvec& All_Fail, 
               vec& All_Risk);

double suplogrank(const uvec& Left_Fail, 
                  const uvec& Left_Risk, 
                  const uvec& All_Fail, 
                  const vec& All_Risk,
                  vec& Temp_Vec);

double CoxGrad(uvec& Pseudo_X,
               const vec& z_eta);


// #############################
// ## Combination Split Trees - NOT IMPLEMENTED##
// #############################

#endif
