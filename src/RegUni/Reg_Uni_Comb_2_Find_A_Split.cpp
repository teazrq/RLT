//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Reg_Uni_Comb_Find_A_Split(Comb_Split_Class& OneSplit,
                               const RLT_REG_DATA& REG_DATA,
                               const PARAM_GLOBAL& Param,
                               const uvec& obs_id,
                               const uvec& var_id,
                               Rand& rngl)
{
  // parameters
  size_t mtry = Param.mtry;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;
  size_t linear_comb = Param.linear_comb;  
  
  // pre-screening variables to get the best ones
  uvec split_var = rngl.sample(var_id, mtry);
  
  //Embedded RF VI Screening Method
  vec split_score = Reg_Uni_Embed_Pre_Screen(REG_DATA,
                                             Param,
                                             obs_id,
                                             split_var,
                                             rngl);

  // sort and get the best ones
  uvec indices = sort_index(split_score, "descend");
  split_var = split_var(indices);

  // if the best variable is categorical
  // do single categorical split
  // I may need to change this later for combination cat split
  if (REG_DATA.Ncat(split_var(0)) > 1)
  {
    size_t j = split_var(0);
    
    // RLTcout << "--Use single cat split" <<  j << std::endl;
    
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    Reg_Uni_Split_Cat(TempSplit, 
                      obs_id, 
                      REG_DATA.X.unsafe_col(j), 
                      REG_DATA.Ncat(j),
                      REG_DATA.Y, 
                      REG_DATA.obsweight, 
                      0.0, // penalty
                      split_gen,
                      1, // univariate splitting rule (var)
                      nsplit,
                      alpha, 
                      useobsweight,
                      rngl);
    
    // record to linear combination split
    OneSplit.var(0) = TempSplit.var;
    OneSplit.load(0) = 1;
    OneSplit.value = TempSplit.value;
    OneSplit.score = TempSplit.score;
    
    return;
  }
  
  // find and restrict to continuous variables at the top
  size_t use_comb = 0;
  
  for (size_t j = 0; j < std::min(linear_comb, mtry); j++)
  {
    if (REG_DATA.Ncat(split_var(j)) == 1)
      use_comb ++;
    else
      break;
  }
  
  //RLTcout << " use combination " << use_comb << std::endl;
  
  split_var.resize(use_comb);
  split_score.resize(use_comb);
  
  // If there is only one continuous variable at the top
  if (use_comb == 1)
  {
    size_t j = split_var(0);
    
    //RLTcout << "--Use single cont split" <<  j << std::endl;
    
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    Reg_Uni_Split_Cont(TempSplit,
                       obs_id,
                       REG_DATA.X.unsafe_col(j), 
                       REG_DATA.Y,
                       REG_DATA.obsweight,
                       0.0, // penalty
                       split_gen,
                       1, // univariate splitting rule (var)
                       nsplit,
                       alpha,
                       useobsweight,
                       rngl);
    
    // record to linear combination split
    OneSplit.var(0) = TempSplit.var;
    OneSplit.load(0) = 1;
    OneSplit.value = TempSplit.value;
    OneSplit.score = TempSplit.score;
    
    return;
  }

  // for more than one variable, find best linear combination split
  Reg_Uni_Comb_Linear(OneSplit,
                      (const uvec&) split_var,
                      REG_DATA,
                      Param,
                      obs_id,
                      rngl);

}