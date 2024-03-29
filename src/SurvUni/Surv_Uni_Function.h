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
          		       imat& ObsTrackPre,
          		       List& param);

void Surv_Uni_Forest_Build(const RLT_SURV_DATA& SURV_DATA,
                           Surv_Uni_Forest_Class& SURV_FOREST,
                           const PARAM_GLOBAL& Param,
                           const uvec& obs_id,
                           const uvec& var_id,
                           imat& ObsTrack,
                           bool do_prediction,
                           mat& Prediction,
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
                        vec& Temp_Vec,
                        const vec& obs_weight,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl);

void Surv_Uni_Split_Cont_Pseudo(Split_Class& TempSplit,
                                const uvec& obs_id,
                                const vec& x,
                                const uvec& Y,
                                const uvec& Censor,
                                const size_t NFail,
                                vec& z_eta,
                                const vec& obs_weight,
                                double penalty,
                                int split_gen,
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

////////////////////
// for prediction 
///////////////////

void Surv_Uni_Forest_Pred(cube& Pred,
                          const Surv_Uni_Forest_Class& SURV_FOREST,
                          const mat& X,
                          const uvec& Ncat,
                          size_t& NFail,
                          size_t usecores,
                          size_t verbose);

////////////////////////////////////////////////////////////
// Survival specific utility functions
////////////////////////////////////////////////////////////

// arrange Y and Censor
void collapse(const uvec& Y, const uvec& Censor, 
              uvec& Y_collapse, uvec& Censor_collapse, 
              uvec& obs_id, size_t& NFail);

// categorical variable arrangement
//Move categorical index
void move_cat_index(size_t& lowindex, 
                    size_t& highindex, 
                    std::vector<Surv_Cat_Class>& cat_reduced, 
                    size_t true_cat, 
                    size_t nmin);

//Record category
double record_cat_split(std::vector<Surv_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat);




////////////////////////////////////////
// splitting score calculations (continuous) old
////////////////////////////////////////

double surv_cont_score_at_cut(const uvec& obs_id,
                             const vec& x,
                             const uvec& Y,
                             const uvec& Censor,
                             const size_t NFail,
                             uvec& All_Fail,
                             vec& All_Risk,
                             vec& Temp_Vec,
                             double a_random_cut,
                             int split_rule);

double surv_cont_score_at_index(uvec& indices,
                                uvec& obs_ranked,
                                const uvec& Y,
                               const uvec& Censor,
                               const size_t NFail,
                               uvec& All_Fail,
                               vec& All_Risk,
                               vec& Temp_Vec,
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
                         vec& Temp_Vec,
                         size_t lowindex, 
                         size_t highindex, 
                         double& temp_cut, 
                         double& temp_score,
                         int split_rule);

// splitting score calculations

double suplogrank(const uvec& Left_Fail, 
                  const uvec& Left_Risk, 
                  const uvec& All_Fail, 
                  const vec& All_Risk,
                  vec& Temp_Vec);

double CoxGrad(uvec& Pseudo_X,
               const vec& z_eta);

// #############################
// ## new splitting functions
// #############################

//// logrank splits 

// calculate logrank scores for all types of split.gen
void Surv_Uni_Logrank_Cont(Split_Class& TempSplit,
                           const uvec& obs_id,
                           const vec& x,
                           const uvec& Y, // Y is collapsed
                           const uvec& Censor, // Censor is collapsed
                           const size_t NFail,
                           const uvec& All_Fail,
                           const uvec& All_Risk, // cumulative
                           int split_gen,
                           int nsplit,
                           double alpha,
                           Rand& rngl);

void Surv_Uni_Logrank_Cat(Split_Class& TempSplit,
                          const uvec& obs_id,
                          const vec& x,
                          const size_t ncat,
                          const uvec& Y, // Y is collapsed
                          const uvec& Censor, // Censor is collapsed
                          const size_t NFail,
                          const uvec& All_Fail,
                          const uvec& All_Risk, // cumulative
                          int split_gen,
                          int nsplit,
                          double alpha,
                          Rand& rngl);

// logrank score at random cut of x value, with pre-calculated at risk and fail
double logrank_at_x_cut(const uvec& obs_id,
                        const vec& x,
                        const uvec& Y, //collapsed
                        const uvec& Censor, //collapsed
                        const size_t NFail,
                        const uvec& All_Fail,
                        const uvec& All_Risk,                        
                        double a_random_cut);

// logrank score at a random index number, provided with sorted index
double logrank_at_id_index(const uvec& indices, // index for Y, sorted by x
                           const uvec& Y, //collapsed
                           const uvec& Censor, //collapsed
                           const size_t NFail,
                           const uvec& All_Fail, 
                           const uvec& All_Risk,
                           size_t a_random_ind);

// logrank test best score 
void logrank_best(const uvec& indices, // index for Y, sorted by x
                  const uvec& obs_id_sorted, // index for x, sorted by x
                  const vec& x, 
                  const uvec& Y, //collapsed
                  const uvec& Censor, //collapsed
                  const size_t NFail,
                  const uvec& All_Fail, 
                  const uvec& All_Risk, 
                  size_t lowindex,
                  size_t highindex,
                  double& temp_cut, 
                  double& temp_score);

// best score for categorical variable
void logrank_best_cat(std::vector<Surv_Cat_Class>& cat_reduced,
                      size_t true_cat,
                      size_t ncat,
                      size_t nmin,
                      size_t N,
                      const uvec& All_Fail,
                      const uvec& All_Risk, // cumulative
                      size_t& best_cat,
                      double& best_score,
                      Rand& rngl);

// general logrank score calculation
double logrank(const uvec& Left_Fail,
               const uvec& Left_Risk,
               const uvec& All_Fail,
               const uvec& All_Risk);


// #############################
// ## Combination Split Trees - NOT IMPLEMENTED##
// #############################

#endif
