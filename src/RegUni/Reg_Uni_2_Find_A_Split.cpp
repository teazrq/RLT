//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Reg_Uni_Find_A_Split(Split_Class& OneSplit,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl)
{
  
  size_t mtry = Param.mtry;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  //bool usevarweight = Param.usevarweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;
  size_t split_rule = Param.split_rule;
  
  // Choose the variables to try
  //mtry = ( (mtry <= var_id.n_elem) ? mtry : var_id.n_elem ); // take minimum

  //uvec sampled = rngl.sample(0, var_id.n_elem - 1, mtry);

  uvec var_try = rngl.sample(var_id, mtry);

  //For each variable in var_try
  for (auto j : var_try)
  {
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;
      
    if (REG_DATA.Ncat(j) > 1) // categorical variable 
    {
      
      Reg_Uni_Split_Cat(TempSplit, 
                        obs_id, 
                        REG_DATA.X.unsafe_col(j), 
                        REG_DATA.Ncat(j),
                        REG_DATA.Y, 
                        REG_DATA.obsweight, 
                        0.0, // penalty
                        split_gen, 
                        split_rule, 
                        nsplit,
                        alpha, 
                        useobsweight,
                        rngl);
      
    }else{ // continuous variable
      
      Reg_Uni_Split_Cont(TempSplit,
                         obs_id,
                         REG_DATA.X.unsafe_col(j), 
                         REG_DATA.Y,
                         REG_DATA.obsweight,
                         0.0, // penalty
                         split_gen,
                         split_rule,
                         nsplit,
                         alpha,
                         useobsweight,
                         rngl);
      
    }
    
    //If this variable is better than the last one tried
    if (TempSplit.score > OneSplit.score)
    {
      //Change to this variable
      OneSplit.var = TempSplit.var;
      OneSplit.value = TempSplit.value;
      OneSplit.score = TempSplit.score;
    }
  }
}