//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Split a node
void Cla_Uni_Split_A_Node(size_t Node,
                          Cla_Uni_Tree_Class& OneTree,
                          const RLT_CLA_DATA& CLA_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;

  // in rf, it is N <= nmin
  if (N <= nmin)
  {
TERMINATENODE:
      Cla_Uni_Terminate_Node(Node, OneTree, obs_id, CLA_DATA.Y, CLA_DATA.nclass, CLA_DATA.obsweight, useobsweight);

  }else{
    RLTcout << " split node " << std::endl;
    
goto TERMINATENODE;

  }
}

// terminate and record a node

void Cla_Uni_Terminate_Node(size_t Node,
                            Cla_Uni_Tree_Class& OneTree,
                            uvec& obs_id,
                            const uvec& Y,
                            const size_t nclass,
                            const vec& obs_weight,
                            bool useobsweight)
{
  
  OneTree.SplitVar(Node) = -1; // -1 mean terminal node. Ow, it would be the variable num

  // calculate node probability
}
