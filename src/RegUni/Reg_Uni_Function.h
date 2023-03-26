//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression Functions
//  **********************************

// my header files
# include "../Utility/Tree_Definition.h"
# include "../Utility/Utility.h"
# include "../Utility/Tree_Function.h"
# include "Reg_Uni_Definition.h"

using namespace Rcpp;
using namespace arma;

#ifndef RLT_REG_UNI_FUNCTION
#define RLT_REG_UNI_FUNCTION

// univariate tree split functions 

List RegUniForestFit(mat& X,
          	         vec& Y,
          		       uvec& Ncat,
          		       vec& obsweight,
          		       vec& varweight,
          		       imat& ObsTrackPre,
          		       List& param);

void Reg_Uni_Forest_Build(const RLT_REG_DATA& REG_DATA,
                          Reg_Uni_Forest_Class& REG_FOREST,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          imat& ObsTrack,
                          bool do_prediction,
                          vec& Prediction,
                          vec& OOBPrediction,
                          vec& VarImp);

void Reg_Uni_Split_A_Node(size_t Node,
                          Reg_Uni_Tree_Class& OneTree,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl);

void Reg_Uni_Split_A_Node_Embed(size_t Node,
                                Reg_Uni_Tree_Class& OneTree,
                                const RLT_REG_DATA& REG_DATA,
                                const PARAM_GLOBAL& Param,
                                uvec& obs_id,
                                const uvec& var_id,
                                const uvec& var_protect,
                                Rand& rngl);
  
void Reg_Uni_Terminate_Node(size_t Node, 
                            Reg_Uni_Tree_Class& OneTree,
                            uvec& obs_id,
                            const vec& Y,
                            const vec& obs_weight,
                            bool useobsweight);


void Reg_Uni_Find_A_Split(Split_Class& OneSplit,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl);

void Reg_Uni_Find_A_Split_Embed(Split_Class& OneSplit,
                                const RLT_REG_DATA& REG_DATA,
                                const PARAM_GLOBAL& Param,
                                const uvec& obs_id,
                                uvec& var_id,
                                uvec& var_protect,
                                Rand& rngl);

void Reg_Uni_Split_Cont(Split_Class& TempSplit,
                        const uvec& obs_id,
                        const vec& x,
                        const vec& Y,
                        const vec& obs_weight,
                        double penalty,
                        size_t split_gen,
                        size_t split_rule,
                        size_t nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl);

void Reg_Uni_Split_Cat(Split_Class& TempSplit,
                       const uvec& obs_id,
                       const vec& x,
                       const size_t ncat,
                       const vec& Y,
                       const vec& obs_weight,
                       double penalty,
                       size_t split_gen,
                       size_t split_rule,
                       size_t nsplit,
                       double alpha,
                       bool useobsweight,
                       Rand& rngl);

// splitting score calculations (continuous)

double reg_uni_cont_score_cut_sub(const uvec& obs_id,
                                  const vec& x,
                                  const vec& Y,
                                  double a_random_cut);

double reg_uni_cont_score_cut_sub_w(const uvec& obs_id,
                                    const vec& x,
                                    const vec& Y,
                                    double a_random_cut,
                                    const vec& obs_weight);

double reg_uni_cont_score_rank_sub(uvec& indices,
                                   const vec& Y,
                                   size_t a_random_ind);

double reg_uni_cont_score_rank_sub_w(uvec& indices,
                                     const vec& Y,
                                     size_t a_random_ind,
                                     const vec& obs_weight);

void reg_uni_cont_score_best_sub(uvec& indices,
                                 const vec& x,
                                 const vec& Y,
                                 size_t lowindex, 
                                 size_t highindex, 
                                 double& temp_cut, 
                                 double& temp_score);


void reg_uni_cont_score_best_sub_w(uvec& indices,
                                   const vec& x,
                                   const vec& Y,
                                   size_t lowindex, 
                                   size_t highindex, 
                                   double& temp_cut, 
                                   double& temp_score,
                                   const vec& obs_weight);

// splitting score calculations (categorical)

double reg_uni_cat_score_cut(std::vector<Reg_Cat_Class>& cat_reduced, 
                             size_t temp_cat,
                             size_t true_cat);

double reg_uni_cat_score_cut_w(std::vector<Reg_Cat_Class>& cat_reduced, 
                               size_t temp_cat,
                               size_t true_cat);

void reg_uni_cat_score_best(std::vector<Reg_Cat_Class>& cat_reduced, 
                            size_t lowindex,
                            size_t highindex,
                            size_t true_cat,
                            size_t& best_cat,
                            double& best_score);

void reg_uni_cat_score_best_w(std::vector<Reg_Cat_Class>& cat_reduced, 
                              size_t lowindex,
                              size_t highindex,
                              size_t true_cat,
                              size_t& best_cat,
                              double& best_score);

// for prediction 

void Reg_Uni_Forest_Pred(mat& Pred,
                         const Reg_Uni_Forest_Class& REG_FOREST,
                         const mat& X,
                         const uvec& Ncat,
                         size_t usecores,
                         size_t verbose);


// #############################
// ## Combination Split Trees ##
// #############################

void Reg_Uni_Comb_Forest_Build(const RLT_REG_DATA& REG_DATA,
                            Reg_Uni_Comb_Forest_Class& REG_FOREST,
                            const PARAM_GLOBAL& Param,
                            const uvec& obs_id,
                            const uvec& var_id,
                            imat& ObsTrack,
                            bool do_prediction,
                            vec& Prediction,
                            vec& OOBPrediction,
                            vec& VarImp);

void Reg_Uni_Comb_Split_A_Node(size_t Node,
                            Reg_Uni_Comb_Tree_Class& OneTree,
                            const RLT_REG_DATA& REG_DATA,
                            const PARAM_GLOBAL& Param,
                            uvec& obs_id,
                            const uvec& var_id,
                            Rand& rngl);

void Reg_Uni_Comb_Terminate_Node(size_t Node,
                              Reg_Uni_Comb_Tree_Class& OneTree,
                              uvec& obs_id,
                              const vec& Y,
                              const vec& obs_weight,
                              bool useobsweight);

void Reg_Uni_Comb_Split_A_Node_Embed(size_t Node,
                                  Reg_Uni_Comb_Tree_Class& OneTree,
                                  const RLT_REG_DATA& REG_DATA,
                                  const PARAM_GLOBAL& Param,
                                  uvec& obs_id,
                                  const uvec& var_id,
                                  const uvec& var_protect,
                                  Rand& rngl);

void Reg_Uni_Comb_Find_A_Split(Comb_Split_Class& OneSplit,
                            const RLT_REG_DATA& REG_DATA,
                            const PARAM_GLOBAL& Param,
                            const uvec& obs_id,
                            const uvec& var_id,
                            Rand& rngl);

void Reg_Uni_Comb_Find_A_Split_Embed(Comb_Split_Class& OneSplit,
                                  const RLT_REG_DATA& REG_DATA,
                                  const PARAM_GLOBAL& Param,
                                  const uvec& obs_id,
                                  const uvec& var_id,
                                  Rand& rngl);

void Reg_Uni_Comb_Pre_Screen(uvec& var,
                             vec& score,
                             const RLT_REG_DATA& REG_DATA, 
                             const PARAM_GLOBAL& Param,
                             const uvec& obs_id,
                             Rand& rngl);
  
void Reg_Uni_Comb_Split_Cont(Comb_Split_Class& OneSplit,
                             const uvec& split_var,
                             const vec& split_score,
                             const RLT_REG_DATA& REG_DATA, 
                             const PARAM_GLOBAL& Param,
                             const uvec& obs_id,
                             Rand& rngl);

double reg_uni_cont_score_cut_full(const vec& xj, 
                                   const vec& y, 
                                   double temp_cut);

double reg_uni_cont_score_cut_full_w(const vec& xj, 
                                     const vec& y,
                                     double temp_cut,
                                     const vec& w);

double reg_uni_cont_score_rank_full(const vec& y,
                                    size_t a_random_ind);
  
double reg_uni_cont_score_rank_full_w(const vec& y,
                                      size_t a_random_ind,
                                      const vec& w);

void reg_uni_cont_score_best_full(const vec& x,
                                  const vec& y,
                                  size_t lowindex, 
                                  size_t highindex, 
                                  double& temp_cut, 
                                  double& temp_score);

void reg_uni_cont_score_best_full_w(const vec& x,
                                    const vec& y,
                                    const vec& w,
                                    size_t lowindex, 
                                    size_t highindex, 
                                    double& temp_cut, 
                                    double& temp_score);

#endif
