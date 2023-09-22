//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header files
# include "../Utility/Tree_Definition.h"
# include "../Utility/Utility.h"
# include "../Utility/Tree_Function.h"
# include "Cla_Uni_Definition.h"

using namespace Rcpp;
using namespace arma;

#ifndef RLT_CLA_UNI_FUNCTION
#define RLT_CLA_UNI_FUNCTION

// univariate tree split functions 

List ClaUniForestFit(arma::mat& X,
                     arma::uvec& Y,
                     arma::uvec& Ncat,
                     size_t nclass,
                     arma::vec& obsweight,
                     arma::vec& varweight,
                     arma::imat& ObsTrack,
                     List& param_r);

void Cla_Uni_Forest_Build(const RLT_CLA_DATA& CLA_DATA,
                          Cla_Uni_Forest_Class& CLA_FOREST,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          imat& ObsTrack,
                          bool do_prediction,
                          mat& Prediction,
                          vec& VarImp);

void Cla_Uni_Split_A_Node(size_t Node,
                          Cla_Uni_Tree_Class& OneTree,
                          const RLT_CLA_DATA& CLA_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl);

void Cla_Uni_Split_A_Node_Embed(size_t Node,
                                Cla_Uni_Tree_Class& OneTree,
                                const RLT_CLA_DATA& CLA_DATA,
                                const PARAM_GLOBAL& Param,
                                uvec& obs_id,
                                const uvec& var_id,
                                const uvec& var_protect,
                                Rand& rngl);

void Cla_Uni_Record_Node(size_t Node,
                         arma::vec& TreeNodeWeight,
                         arma::mat& TreeNodeProb,
                         uvec& obs_id,
                         const uvec& Y,
                         const size_t nclass,
                         const vec& obs_weight,
                         bool useobsweight);

void Cla_Uni_Find_A_Split(Split_Class& OneSplit,
                          const RLT_CLA_DATA& Cla_DATA,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl);

void Cla_Uni_Find_A_Split_Embed(Split_Class& OneSplit,
                                const RLT_CLA_DATA& Cla_DATA,
                                const PARAM_GLOBAL& Param,
                                const uvec& obs_id,
                                uvec& var_id,
                                uvec& var_protect,
                                Rand& rngl);

void Cla_Uni_Split_Cat(Split_Class& TempSplit,
                       const uvec& obs_id,
                       const vec& x,
                       const size_t ncat,
                       const uvec& Y,
                       const vec& obs_weight,
                       const size_t nclass,
                       double penalty,
                       size_t split_gen,
                       size_t split_rule,
                       size_t nsplit,
                       double alpha,
                       bool useobsweight,
                       Rand& rngl);

void Cla_Uni_Split_Cont(Split_Class& TempSplit,
                        const uvec& obs_id,
                        const vec& x,
                        const uvec& Y,
                        const vec& obs_weight,
                        const size_t nclass,
                        double penalty,
                        size_t split_gen,
                        size_t split_rule,
                        size_t nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl);

// splitting score functions 

double cla_uni_cont_score_cut_sub(const uvec& obs_id,
                                  const vec& x,
                                  const uvec& Y,
                                  size_t nclass,
                                  double a_random_cut);

double cla_uni_cont_score_cut_sub_w(const uvec& obs_id,
                                    const vec& x,
                                    const uvec& Y,
                                    size_t nclass,
                                    double a_random_cut,
                                    const vec& obs_weight);

double cla_uni_cont_score_rank_sub(uvec& indices,
                                   const uvec& Y,
                                   size_t nclass,
                                   size_t a_random_ind);

double cla_uni_cont_score_rank_sub_w(uvec& indices,
                                     const uvec& Y,
                                     size_t nclass,
                                     size_t a_random_ind,
                                     const vec& obs_weight);

void cla_uni_cont_score_best_sub(uvec& indices,
                                 const vec& x,
                                 const uvec& Y,
                                 size_t nclass,
                                 size_t lowindex, 
                                 size_t highindex, 
                                 double& temp_cut, 
                                 double& temp_score);

void cla_uni_cont_score_best_sub_w(uvec& indices,
                                   const vec& x,
                                   const uvec& Y,
                                   size_t nclass,
                                   size_t lowindex, 
                                   size_t highindex, 
                                   double& temp_cut, 
                                   double& temp_score,
                                   const vec& obs_weight);

double cla_uni_cat_score_cut(std::vector<Cla_Cat_Class>& cat_reduced, 
                             size_t temp_cat, 
                             size_t true_cat);

void cla_uni_cat_score_best(std::vector<Cla_Cat_Class>& cat_reduced,
                            size_t true_cat,
                            size_t ncat,
                            size_t nmin,
                            size_t& best_cat,
                            double& best_score,
                            Rand& rngl);

void cla_uni_cat_score_best_large(std::vector<Cla_Cat_Class>& cat_reduced,
                                  size_t true_cat,
                                  size_t ncat,
                                  size_t nmin,
                                  size_t& best_cat,
                                  double& best_score,
                                  Rand& rngl);

// categorical variable arranging 
//Move categorical index
void move_cat_index(size_t& lowindex, 
                    size_t& highindex, 
                    std::vector<Cla_Cat_Class>& cat_reduced, 
                    size_t true_cat, 
                    size_t nmin);

//Record category
double record_cat_split(std::vector<Cla_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat);


// #############################
// ## Combination Split Trees ##
// #############################

vec Cla_Uni_Embed_Pre_Screen(const RLT_CLA_DATA& Cla_DATA,
                             const PARAM_GLOBAL& Param,
                             const uvec& obs_id,
                             const uvec& var_id,
                             Rand& rngl);

void Cla_Uni_Comb_Forest_Build(const RLT_CLA_DATA& CLA_DATA,
                               Cla_Uni_Comb_Forest_Class& CLA_FOREST,
                               const PARAM_GLOBAL& Param,
                               const uvec& obs_id,
                               const uvec& var_id,
                               imat& ObsTrack,
                               bool do_prediction,
                               mat& Prediction,
                               vec& VarImp);

void Cla_Uni_Comb_Split_A_Node(size_t Node,
                               Cla_Uni_Comb_Tree_Class& OneTree,
                               const RLT_CLA_DATA& CLA_DATA,
                               const PARAM_GLOBAL& Param,
                               uvec& obs_id,
                               const uvec& var_id,
                               const uvec& var_protect,
                               Rand& rngl);

void Cla_Uni_Comb_Find_A_Split(Comb_Split_Class& OneSplit,
                               const RLT_CLA_DATA& Cla_DATA,
                               const PARAM_GLOBAL& Param,
                               const uvec& obs_id,
                               uvec& var_id,
                               uvec& var_protect,
                               Rand& rngl);

#endif
