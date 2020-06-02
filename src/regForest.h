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

List RegForestUniFit(mat& X,
          	         vec& Y,
          		     uvec& Ncat,
          		     List& param,
          		     List& RLTparam,
          		     vec& obsweight,
          		     vec& varweight,
          		     int usecores,
          		     int verbose,
                     umat& ObsTrackPre);

void Reg_Uni_Forest_Build(const RLT_REG_DATA& REG_DATA,
                          Reg_Uni_Forest_Class& REG_FOREST,
                          const PARAM_GLOBAL& Param,
                          const PARAM_RLT& Param_RLT,
                          uvec& obs_id,
                          uvec& var_id,
                          umat& ObsTrack,
                          vec& Prediction,
                          vec& OOBPrediction,
                          vec& VarImp,
                          size_t seed,
                          int usecores,
                          int verbose);

void Reg_Uni_Split_A_Node(size_t Node,
                          Reg_Uni_Tree_Class& OneTree,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          const PARAM_RLT& Param_RLT,
                          uvec& obs_id,
                          uvec& var_id);

void Reg_Uni_Terminate_Node(size_t Node, 
                            Reg_Uni_Tree_Class& OneTree,
                            uvec& obs_id,
                            const vec& Y,
                            const vec& obs_weight,
                            const PARAM_GLOBAL& Param,
                            bool useobsweight);


void Reg_Uni_Find_A_Split(Uni_Split_Class& OneSplit,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          const PARAM_RLT& RLTParam,
                          uvec& obs_id,
                          uvec& var_id);

void Reg_Uni_Find_A_Split_Embed(Uni_Split_Class& OneSplit,
                                const RLT_REG_DATA& REG_DATA,
                                const PARAM_GLOBAL& Param,
                                const PARAM_RLT& RLTParam,
                                uvec& obs_id,
                                uvec& var_id);

void Reg_Uni_Split_Cont(Uni_Split_Class& TempSplit,
                        uvec& obs_id,
                        const vec& x,
                        const vec& Y,
                        const vec& obs_weight,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        size_t nmin,
                        double alpha,
                        bool useobsweight);

void Reg_Uni_Split_Cat(Uni_Split_Class& TempSplit,
                       uvec& obs_id,
                       const vec& x,
                       const size_t ncat,
                       const vec& Y,
                       const vec& obs_weight,
                       double penalty,
                       int split_gen,
                       int split_rule,
                       int nsplit,
                       size_t nmin,
                       double alpha,
                       bool useobsweight);

// splitting score calculations (continuous)

double reg_cont_score_at_cut(uvec& obs_id,
                            const vec& x,
                            const vec& Y,
                            double a_random_cut);

double reg_cont_score_at_cut_w(uvec& obs_id,
                              const vec& x,
                              const vec& Y,
                              double a_random_cut,
                              const vec& obs_weight);

double reg_cont_score_at_index(uvec& indices,
                              const vec& Y,
                              size_t a_random_ind);

double reg_cont_score_at_index_w(uvec& indices,
                                const vec& Y,
                                size_t a_random_ind,
                                const vec& obs_weight);

void reg_cont_score_best(uvec& indices,
                        const vec& x,
                        const vec& Y,
                        size_t lowindex, 
                        size_t highindex, 
                        double& temp_cut, 
                        double& temp_score);

void reg_cont_score_best_w(uvec& indices,
                          const vec& x,
                          const vec& Y,
                          size_t lowindex, 
                          size_t highindex, 
                          double& temp_cut, 
                          double& temp_score,
                          const vec& obs_weight);

// splitting score calculations (categorical)

double reg_cat_score(std::vector<Reg_Cat_Class>& cat_reduced, 
                     size_t temp_cat, 
                     size_t true_cat);

double reg_cat_score_w(std::vector<Reg_Cat_Class>& cat_reduced, 
                       size_t temp_cat, 
                       size_t true_cat);

void reg_cat_score_best(std::vector<Reg_Cat_Class>& cat_reduced, 
                        size_t lowindex,
                        size_t highindex,
                        size_t true_cat,
                        size_t& best_cat,
                        double& best_score);

void reg_cat_score_best_w(std::vector<Reg_Cat_Class>& cat_reduced, 
                          size_t lowindex,
                          size_t highindex,
                          size_t true_cat,
                          size_t& best_cat,
                          double& best_score);

// other utilities functions for regression 
/*
void reg_move_cat_index(size_t& lowindex, 
        		size_t& highindex, 
        		std::vector<Reg_Cat_Class>& cat_reduced, 
        		size_t true_cat, 
        		size_t nmin);

bool reg_cat_reduced_compare(Reg_Cat_Class& a, 
                             Reg_Cat_Class& b);
*/
// for prediction 

void Reg_Uni_Forest_Pred(mat& Pred,
                         const Reg_Uni_Forest_Class& REG_FOREST,
                         const mat& X,
                         const uvec& Ncat,
                         const uvec& treeindex,
                         int usecores,
                         int verbose);

#endif
