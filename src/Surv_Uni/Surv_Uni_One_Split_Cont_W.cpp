//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Split_Cont_W(Uni_Split_Class& TempSplit, 
                             uvec& obs_id,
                             const vec& x,
                             const uvec& Y, // Y is collapsed
                             const uvec& Censor, // Censor is collapsed
                             const vec& obs_weight,
                             size_t NFail,
                             double penalty,
                             int split_gen,
                             int split_rule,
                             int nsplit,
                             size_t nmin, 
                             double alpha,
                             bool failforce)
{
    Rcout << "      --- Weighted Splitting Rule (continuous) for Survival not Implemented Yet" << std::endl;
}