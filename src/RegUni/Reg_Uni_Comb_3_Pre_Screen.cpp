//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Screening method
void Reg_Uni_Comb_Pre_Screen(uvec& var,
                             vec& score,
                             const RLT_REG_DATA& REG_DATA, 
                             const PARAM_GLOBAL& Param,
                             const uvec& obs_id,
                             Rand& rngl)
{
  bool useobsweight = Param.useobsweight;
  
  //explore each variable in var_try
  for (size_t j = 0; j < var.n_elem; j++)
  {
    size_t var_j = var(j);
    
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = var_j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    if (REG_DATA.Ncat(var_j) > 1) // categorical variable 
    {
      
      Reg_Uni_Split_Cat(TempSplit, 
                        obs_id, 
                        REG_DATA.X.unsafe_col(var_j), 
                        REG_DATA.Ncat(var_j),
                        REG_DATA.Y, 
                        REG_DATA.obsweight, 
                        0.0, // penalty
                        3, // use best split
                        1, // splitting rule var (not used in function)
                        0, // nsplit
                        0.0, // alpha
                        useobsweight,
                        rngl);
      
    }else{ // continuous variable
      
      Reg_Uni_Split_Cont(TempSplit,
                         obs_id,
                         REG_DATA.X.unsafe_col(var_j), 
                         REG_DATA.Y,
                         REG_DATA.obsweight,
                         0.0, // penalty
                         3, // use best split
                         1, // splitting rule var (not used in function)
                         0, // nsplit
                         0.0, // alpha
                         useobsweight,
                         rngl);
    }
    
    score(j) = TempSplit.score;
    
  }
}
