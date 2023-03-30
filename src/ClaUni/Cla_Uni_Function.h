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
                          mat& OOBPrediction,
                          vec& VarImp);

void Cla_Uni_Split_A_Node(size_t Node,
                          Cla_Uni_Tree_Class& OneTree,
                          const RLT_CLA_DATA& CLA_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl);

void Cla_Uni_Terminate_Node(size_t Node,
                            Cla_Uni_Tree_Class& OneTree,
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

// for prediction 

void Cla_Uni_Forest_Pred(cube& Pred,
                         const Cla_Uni_Forest_Class& CLA_FOREST,
                         const mat& X,
                         const uvec& Ncat,
                         size_t usecores,
                         size_t verbose);

#endif
