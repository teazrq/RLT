//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../regForest.h"

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Find_A_Split_Embed(Uni_Split_Class& OneSplit,
                                const RLT_REG_DATA& REG_DATA,
                                const PARAM_GLOBAL& Param,
                                const PARAM_RLT& RLTParam,
                                uvec& obs_id,
                                uvec& var_id)
{
  
  Rcout << "    --- Reg_Find_A_Split with embedded model " << std::endl;
  
  PARAM_GLOBAL Embed_Param = Param;

  Embed_Param.print();
  
  
  
}

