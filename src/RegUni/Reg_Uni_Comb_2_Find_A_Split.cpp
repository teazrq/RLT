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
  
  RLTcout << "Reg_Uni_Comb_Find_A_Split:" << std::endl;
  
  // preset
  OneSplit.var.zeros();
  OneSplit.load.zeros();
  OneSplit.score = -1;
  OneSplit.value = 0;
  size_t linear_comb = Param.linear_comb;
  
  // parameters
  size_t mtry = Param.mtry;
  //size_t nmin = Param.nmin;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  //bool usevarweight = Param.usevarweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;

  // pre-screening all variables to get the best ones
  uvec split_var = rngl.sample(var_id, mtry);
  vec split_score(split_var.n_elem, fill::zeros);
  
  Reg_Uni_Comb_Pre_Screen(split_var,
                          split_score,
                          REG_DATA,
                          Param,
                          obs_id,
                          rngl);
  
  // sort and get the best ones
  uvec indices = sort_index(split_score, "descend");
  split_var = split_var(indices);
  split_score = split_score(indices);
  //split_value = split_value(indices);
  
  RLTcout << "best vars \n" << split_var << std::endl;
  RLTcout << "best scores \n" << split_score << std::endl;

  // if the best variable is categorical
  // do single categorical split
  // I may need to change this later for combination cat split
  if (REG_DATA.Ncat(split_var(0)) > 1)
  {
    size_t var_j = split_var(0);
    
    RLTcout << "Use single cat split" <<  var_j << std::endl;
    
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = var_j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    Reg_Uni_Split_Cat(TempSplit, 
                      obs_id, 
                      REG_DATA.X.unsafe_col(var_j), 
                      REG_DATA.Ncat(var_j),
                      REG_DATA.Y, 
                      REG_DATA.obsweight, 
                      0.0, // penalty
                      split_gen,
                      1, // splitting rule var (not used in function)
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
  
  // find all continuous variables at the top
  
  size_t cont_count = 0;
  uvec use_var(linear_comb, fill::zeros);
  
  for (size_t j = 0; j < linear_comb; j++)
  {
    if (REG_DATA.Ncat(split_var(j)) == 1)
    {
      use_var(cont_count) = split_var(j);
      cont_count++;
    }
  }
  // If there is only one continuous variable at the top
  if (cont_count == 1)
  {
    size_t var_j = split_var(0);
    
    RLTcout << "Use single cont split" <<  var_j << std::endl;
    
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = var_j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    Reg_Uni_Split_Cont(TempSplit,
                       obs_id,
                       REG_DATA.X.unsafe_col(var_j), 
                       REG_DATA.Y,
                       REG_DATA.obsweight,
                       0.0, // penalty
                       split_gen,
                       1, // splitting rule: var (not used in function)
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
  
  // continuous variable linear combination 
  // use all continuous variables in the top
  
  use_var.resize(cont_count);
  
  // find best linear combination split
  Reg_Uni_Comb_Split_Cont(OneSplit,
                          (const uvec&) use_var,
                          REG_DATA,
                          Param,
                          obs_id,
                          rngl);

  return;
}