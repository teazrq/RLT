//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Cla_Uni_Find_A_Split(Split_Class& OneSplit,
                          const RLT_CLA_DATA& Cla_DATA,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl)
{
  
  size_t mtry = Param.mtry;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;
  size_t split_rule = Param.split_rule;
  
  // sample variables for mtry
  uvec var_try = rngl.sample(var_id, mtry);

  //For each variable in var_try
  for (auto j : var_try)
  {
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;
      
    if (Cla_DATA.Ncat(j) > 1) // categorical variable 
    {
      
      Cla_Uni_Split_Cat(TempSplit, 
                        obs_id, 
                        Cla_DATA.X.unsafe_col(j), 
                        Cla_DATA.Ncat(j),
                        Cla_DATA.Y, 
                        Cla_DATA.obsweight,
                        Cla_DATA.nclass,
                        0.0, // penalty
                        split_gen, 
                        split_rule, 
                        nsplit,
                        alpha, 
                        useobsweight,
                        rngl);
      
    }else{ // continuous variable
      
      Cla_Uni_Split_Cont(TempSplit,
                         obs_id,
                         Cla_DATA.X.unsafe_col(j), 
                         Cla_DATA.Y,
                         Cla_DATA.obsweight,
                         Cla_DATA.nclass,
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