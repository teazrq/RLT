//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../claForest.h"

#include <xoshiro.h>
#include <dqrng_distribution.h>

using namespace Rcpp;
using namespace arma;

void Cla_Multi_Forest_Build(const RLT_REG_DATA& REG_DATA,
                            Reg_Uni_Forest_Class& REG_FOREST,
                          const PARAM_GLOBAL& Param,
                          const PARAM_RLT& Param_RLT,
                          uvec& obs_id,
                          uvec& var_id,
                          umat& ObsTrack,
                          vec& Prediction,
                          vec& OOBPrediction,
                          vec& VarImp,
                          size_t seed, // this is not done yet
                          int usecores,
                          int verbose)
{

  
}