//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Comb_Find_A_Split_Embed(Comb_Split_Class& OneSplit,
                                  const RLT_REG_DATA& REG_DATA,
                                  const PARAM_GLOBAL& Param,
                                  const uvec& obs_id,
                                  const uvec& var_id,
                                  Rand& rngl)
{
  
  Rcout << "Reg_Uni_Comb_Find_A_Split_Embed" << std::endl;
  
  PARAM_GLOBAL Embed_Param = Param;
  
  Embed_Param.print();
}

