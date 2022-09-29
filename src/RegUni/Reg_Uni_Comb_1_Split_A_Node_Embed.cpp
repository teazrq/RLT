//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Split a node
void Reg_Uni_Comb_Split_A_Node_Embed(size_t Node,
                                  Reg_Uni_Comb_Tree_Class& OneTree,
                                  const RLT_REG_DATA& REG_DATA,
                                  const PARAM_GLOBAL& Param,
                                  uvec& obs_id,
                                  const uvec& var_id,
                                  const uvec& var_protect,
                                  Rand& rngl)
{
  RLTcout << "Reg_Uni_Comb_Split_A_Node_Embed" << std::endl;
}
