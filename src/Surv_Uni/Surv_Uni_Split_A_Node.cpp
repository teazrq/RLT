//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Split_A_Node(size_t Node,
                           Surv_Uni_Tree_Class& OneTree,
                           std::vector<arma::uvec>& OneNodeRegi,
                           const mat& X,
                           const uvec& Y,
                           const uvec& Censor,
                           const uvec& Ncat,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& Param_RLT,
                           vec& obs_weight,
                           uvec& obs_id,
                           vec& var_weight,
                           uvec& var_id)
{

  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;
  size_t N = obs_id.n_elem;
  bool kernel_ready = Param.kernel_ready;
    
  // calculate node information
  DEBUG_Rcout << "  -- Reg_Split_A_Node on Node " << Node << " with sample size " << obs_id.size() << std::endl;
  
  if (N < 2*nmin)
  {
TERMINATENODE:

    DEBUG_Rcout << "  -- Terminate node " << Node << std::endl;
    Surv_Uni_Terminate_Node(Node, OneTree, OneNodeRegi, Y, Censor, Param, obs_weight, obs_id, useobsweight);
    
  }else{
    
    DEBUG_Rcout << "  -- find a split " << Node << std::endl;
    goto TERMINATENODE;
  }
}

// terminate and record a node

void Surv_Uni_Terminate_Node(size_t Node, 
                             Surv_Uni_Tree_Class& OneTree,
                             std::vector<arma::uvec>& OneNodeRegi,
                             const uvec& Y,
                             const uvec& Censor,
                             const PARAM_GLOBAL& Param,
                             vec& obs_weight,
                             uvec& obs_id,
                             bool useobsweight)
{
  
  OneTree.NodeType(Node) = 3; // 0: unused, 1: reserved; 2: internal node; 3: terminal node
  OneTree.NodeSize(Node) = obs_id.n_elem;
  
  if (Param.kernel_ready)
    OneNodeRegi[Node] = uvec(&obs_id[0], obs_id.n_elem, false, true);
  
  if (useobsweight)
  {
    // DEBUG_Rcout << "terminate weighted" << std::endl;
    
  }else{
    // DEBUG_Rcout << "terminate nonweighted" << std::endl;
    
  }

}
