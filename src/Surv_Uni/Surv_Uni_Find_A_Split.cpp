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
  DEBUG_Rcout << "    --- Surv_Uni_Find_A_Split " << std::endl;
  
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
  
  // sort Y values 
  
  std::sort(obs_id.begin(), obs_id.end(), [&Y, &Censor](size_t i, size_t j)
  {
      if (Y(i) == Y(j))
          return(Censor(i) > Censor(j));
      else
          return Y(i) < Y(j);
  });
  
  // collapse Y into contiguous integers 
  size_t NFail;
  uvec Y_collapse(N);
  uvec Censor_collapse(N);
  
  DEBUG_Rcout << "    --- Y before collapse \n" << join_rows(Y(obs_id), Censor(obs_id))  << std::endl;
 
  collapse(Y, Censor, Y_collapse, Censor_collapse, obs_id, NFail);
  
  DEBUG_Rcout << "    --- Y after collapse " << std::endl;
  DEBUG_Rcout << join_rows(Y_collapse, Censor_collapse) << std::endl;
  DEBUG_Rcout << "    --- number of failure " << NFail << std::endl;
  
  if (NFail == 0)
    return; 
  
  // start univariate search

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
      Surv_Uni_Split_Cat(TempSplit, obs_id, X.unsafe_col(temp_var), Y_collapse, Censor_collapse, var_weight(temp_var), 
                         split_gen, split_rule, nsplit, nmin, alpha, obs_weight, useobsweight, NFail, 0, Ncat(temp_var));
        
        DEBUG_Rcout << "      --- try var " << temp_var << " at cut " << TempSplit.value << " (categorical) with score " << TempSplit.score << std::endl;
      
    }else{ // continuous variable
      
      Surv_Uni_Split_Cont(TempSplit, obs_id, X.unsafe_col(temp_var), Y_collapse, Censor_collapse, var_weight(temp_var),
                          split_gen, split_rule, nsplit, nmin, alpha, obs_weight, useobsweight, NFail, 0);
        
        DEBUG_Rcout << "      --- try var " << temp_var << " at cut " << TempSplit.value << " (continuous) with score " << TempSplit.score << std::endl;
    }
    
    if (TempSplit.score > OneSplit.score)
    {
      OneSplit.var = TempSplit.var;
      OneSplit.value = TempSplit.value;
      OneSplit.score = TempSplit.score;
    }
  }
}







// collapse Y into contiguous integers, Y will always be in an increasing order, failure observations come first

void collapse(const uvec& Y, const uvec& Censor, uvec& Y_collapse, uvec& Censor_collapse, uvec& obs_id, size_t& NFail)
{
    size_t N = obs_id.n_elem;
    
    size_t timepoint = 0;
    size_t current_y = Y[obs_id[0]];
    size_t i = 0;
    
    if (Censor[obs_id[0]] == 1) // no censoring before first failure
    {
        timepoint = 1;
        Y_collapse[0] = 1;
        Censor_collapse[0] = 1;
        i = 1;
    }else{
        while(i < N and Censor[obs_id[i]] == 0)
        {
            Y_collapse[i] = timepoint;
            Censor_collapse[i] = 0;
            current_y = Y[obs_id[i]];
            i++;
            
        }
    }
    
    for(; i < N; i++)
    {
        // a new failure point, update new Y
        if (Y[obs_id[i]] > current_y and Censor[obs_id[i]] == 1)
        {
            timepoint ++;
            current_y = Y[obs_id[i]];
        }
        
        // otherwise, Y_collapse is just the current timepoint
        Y_collapse[i] = timepoint;
        Censor_collapse[i] = Censor[obs_id[i]];
    }
    
    NFail = timepoint;
}
