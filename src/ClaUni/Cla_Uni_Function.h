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

// for prediction 

void Cla_Uni_Forest_Pred(mat& Pred,
                         const Cla_Uni_Forest_Class& CLA_FOREST,
                         const mat& X,
                         const uvec& Ncat,
                         size_t usecores,
                         size_t verbose);

#endif
