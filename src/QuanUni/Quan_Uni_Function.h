//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression Functions
//  **********************************

// my header files
# include "../Utility/Tree_Definition.h"
# include "../Utility/Utility.h"
# include "../Utility/Tree_Function.h"
# include "Quan_Uni_Definition.h"

using namespace Rcpp;
using namespace arma;

#ifndef RLT_QUAN_UNI_FUNCTION
#define RLT_QUAN_UNI_FUNCTION

// univariate tree split functions 

List QuanUniForestFit(mat& X,
          	         vec& Y,
          		       uvec& Ncat,
          		       vec& obsweight,
          		       vec& varweight,
          		       imat& ObsTrackPre,
          		       List& param);

void Quan_Uni_Forest_Build(const RLT_REG_DATA& REG_DATA,
                          Reg_Uni_Forest_Class& REG_FOREST,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          imat& ObsTrack,
                          bool do_prediction,
                          vec& Prediction,
                          vec& OOBPrediction,
                          vec& VarImp);

void Quan_Uni_Split_A_Node(size_t Node,
                           Reg_Uni_Tree_Class& OneTree,
                           const RLT_REG_DATA& REG_DATA,
                           const PARAM_GLOBAL& Param,
                           uvec& obs_id,
                           const uvec& var_id,
                           Rand& rngl);

void Quan_Uni_Find_A_Split(Split_Class& OneSplit,
                           const RLT_REG_DATA& REG_DATA,
                           const PARAM_GLOBAL& Param,
                           const uvec& obs_id,
                           const uvec& var_id,
                           Rand& rngl);

void Quan_Uni_Split_Cont(Split_Class& TempSplit,
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


//Calculate a KS score at a random cut
double quan_uni_cont_score_cut_sub(const uvec& obs_id,
                                   const vec& x,
                                   const vec& Y,
                                   double a_random_cut);
#endif
