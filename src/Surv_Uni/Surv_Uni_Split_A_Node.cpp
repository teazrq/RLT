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
                           const size_t NFail,
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
    Surv_Uni_Terminate_Node(Node, OneTree, OneNodeRegi, Y, Censor, NFail, Param, obs_weight, obs_id, useobsweight);
    
  }else{
    
    Uni_Split_Class OneSplit;
    
    Surv_Uni_Find_A_Split(OneSplit, X, Y, Censor, Ncat, Param, Param_RLT, obs_weight, obs_id, var_weight, var_id);
    
    DEBUG_Rcout << "  -- Found split on variable " << OneSplit.var << " cut " << OneSplit.value << " and score " << OneSplit.score << std::endl;
    
    // if did not find a good split, terminate
    if (OneSplit.score <= 0)
      goto TERMINATENODE;
    
    // construct indices for left and right nodes
    DEBUG_Rcout << "  -- splitting value is " << OneSplit.value << std::endl;
    
    uvec left_id(obs_id.n_elem);
    
    if ( Ncat(OneSplit.var) == 1 )
    {
      split_id(X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id);  
      
      DEBUG_Rcout << "  -- select cont variable " << OneSplit.var << " split at " << OneSplit.value << std::endl;
    }else{
      split_id_cat(X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id, Ncat(OneSplit.var));
      
      DEBUG_Rcout << "  -- select cat variable " << OneSplit.var << " split at " << OneSplit.value << std::endl;
    }    
    
    // if this happens something about the splitting rule is wrong
    if (left_id.n_elem == N or obs_id.n_elem == N)
      goto TERMINATENODE;
    
    // check if the current tree is long enough to store two more nodes
    // if not, extend the current tree
    
    if ( OneTree.NodeType( OneTree.NodeType.size() - 2) > 0 )
    {
      DEBUG_Rcout << "  ------------- extend tree length: this shouldn't happen ----------- " << std::endl;
      
      // extend tree structure
      OneTree.extend();
      
      // extend noderegi
      if ( kernel_ready and (OneTree.NodeType.n_elem > OneNodeRegi.size()) )
        OneNodeRegi.resize( OneTree.NodeType.n_elem ); // I think this creates copy, we need a more efficent way to do it...    
    }    
    
    // find the locations of next left and right nodes     
    OneTree.NodeType(Node) = 2; // 0: unused, 1: reserved; 2: internal node; 3: terminal node	
    size_t NextLeft = Node;
    size_t NextRight = Node;
    OneTree.find_next_nodes(NextLeft, NextRight);
    
    DEBUG_Rcout << "  -- Next Left at " << NextLeft << std::endl;
    DEBUG_Rcout << "  -- Next Right at " << NextRight << std::endl;

    // record tree 
    
    OneTree.SplitVar(Node) = OneSplit.var;
    OneTree.SplitValue(Node) = OneSplit.value;
    OneTree.LeftNode(Node) = NextLeft;
    OneTree.RightNode(Node) = NextRight;
    
    OneTree.NodeSize(Node) = left_id.n_elem + obs_id.n_elem;
    

    Surv_Uni_Split_A_Node(NextLeft, OneTree, OneNodeRegi,
                          X, Y, Censor, NFail, Ncat, Param, Param_RLT,
                          obs_weight, left_id, var_weight, var_id);
    
    
    Surv_Uni_Split_A_Node(NextRight, OneTree, OneNodeRegi,
                          X, Y, Censor, NFail, Ncat, Param, Param_RLT,
                          obs_weight, obs_id, var_weight, var_id);

  }
}

// terminate and record a node

void Surv_Uni_Terminate_Node(size_t Node, 
                             Surv_Uni_Tree_Class& OneTree,
                             std::vector<arma::uvec>& OneNodeRegi,
                             const uvec& Y,
                             const uvec& Censor,
                             const size_t NFail,
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
    
    OneTree.NodeHaz(Node) = {1, 0.8, 0.6, 0.4, 0.2, 0}; // replace later 
    
  }else{
    // DEBUG_Rcout << "terminate nonweighted" << std::endl;
    
    OneTree.NodeHaz(Node).zeros(NFail + 1); // replace later 
    OneTree.NodeHaz(Node)(0) = Node; // this one is to backtrack node ID, there should not be any failure here
    
    uvec NodeCensor(NFail + 1, fill::zeros);
    
    for (size_t i = 0; i < obs_id.n_elem; i++)
    {
      if (Censor(obs_id(i)) == 0)
        NodeCensor( Y(obs_id(i)) )++;
      else
        OneTree.NodeHaz(Node)( Y(obs_id(i)) )++;
    }
    
    size_t N = obs_id.n_elem - NodeCensor(0);
    double h = 1;
    
    for (size_t j = 1; j < NFail + 1; j++)
    {
      if (N <= 0) break;
        
      h = OneTree.NodeHaz(Node)(j) / N;
      N -= OneTree.NodeHaz(Node)(j) + NodeCensor(j);
      OneTree.NodeHaz(Node)(j) = h;
    }
      
    DEBUG_Rcout << "node Y" << join_rows(Y(obs_id), Censor(obs_id)) << std::endl;
    DEBUG_Rcout << "node surv" << OneTree.NodeHaz(Node) << std::endl;
    
    
  }

}
