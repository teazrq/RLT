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

void Surv_Uni_Split_Cont(Uni_Split_Class& TempSplit, 
                        uvec& obs_id,
                        const vec& x,
                        const uvec& Y,
                        const uvec& Censor,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        size_t nmin, 
                        double alpha,
                        vec& obs_weight,
                        bool useobsweight)
{
  DEBUG_Rcout << "        --- Surv_One_Split_Cont" << std::endl;
}