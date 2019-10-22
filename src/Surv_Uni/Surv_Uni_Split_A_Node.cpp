//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
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
    
    Uni_Split_Class OneSplit;
    
    Surv_Uni_Find_A_Split(OneSplit, X, Y, Censor, Ncat, Param, Param_RLT, obs_weight, obs_id, var_weight, var_id);
    
    DEBUG_Rcout << "  -- Found split on variable " << OneSplit.var << " cut " << OneSplit.value << " and score " << OneSplit.score << std::endl;
    
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
    
    OneTree.NodeSurv(Node) = {1, 0.8, 0.6, 0.4, 0.2, 0}; // replace later 
    
  }else{
    // DEBUG_Rcout << "terminate nonweighted" << std::endl;
    
    OneTree.NodeSurv(Node) = {1, 0.9, 0.7, 0.5, 0.3, 0.1, 0}; // replace later 
  }

}
