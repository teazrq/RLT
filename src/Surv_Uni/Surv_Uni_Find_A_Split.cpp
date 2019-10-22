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

void Surv_Uni_Find_A_Split(Uni_Split_Class& OneSplit,
						  const mat& X,
						  const uvec& Y,
						  const uvec& Censor,
						  const uvec& Ncat,
						  const PARAM_GLOBAL& Param,
						  const PARAM_RLT& RLTParam,
						  vec& obs_weight,
						  uvec& obs_id,
						  vec& var_weight,
						  uvec& var_id)
{
  
  size_t mtry = Param.mtry;
  size_t nmin = Param.nmin;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  bool usevarweight = Param.usevarweight;
  int nsplit = Param.nsplit;
  int split_gen = Param.split_gen;
  int split_rule = Param.split_rule;
  bool reinforcement = Param.reinforcement;
  
  size_t N = obs_id.n_elem;
  size_t P = var_id.n_elem;

  mtry = ( (mtry <= P) ? mtry:P ); // take minimum
  
  uvec var_try = arma::randperm(P, mtry);
  
  DEBUG_Rcout << "    --- Surv_Find_A_Split with mtry = " << mtry << std::endl;

  for (size_t j = 0; j < mtry; j++)
  {
    size_t temp_var = var_id(var_try(j));
    
    Uni_Split_Class TempSplit;
    TempSplit.var = temp_var;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    DEBUG_Rcout << "    --- try var " << temp_var << std::endl;
      
    if (Ncat(temp_var) > 1) // categorical variable 
    {
      Surv_Uni_Split_Cat(TempSplit, obs_id, X.unsafe_col(temp_var), Y, Censor, 0.0, 
                         split_gen, split_rule, nsplit, nmin, alpha, obs_weight, useobsweight, Ncat(temp_var));
        
      DEBUG_Rcout << "    --- try var " << temp_var << " at cut " << TempSplit.value << " (categorical) with score " << TempSplit.score << std::endl;
      
    }else{ // continuous variable
      
      Surv_Uni_Split_Cont(TempSplit, obs_id, X.unsafe_col(temp_var), Y, Censor, 0.0, 
                          split_gen, split_rule, nsplit, nmin, alpha, obs_weight, useobsweight);
        
      DEBUG_Rcout << "    --- try var " << temp_var << " at cut " << TempSplit.value << " (continuous) with score " << TempSplit.score << std::endl;
    }
    
    if (TempSplit.score > OneSplit.score)
    {
      OneSplit.var = TempSplit.var;
      OneSplit.value = TempSplit.value;
      OneSplit.score = TempSplit.score;
    }
  }
}