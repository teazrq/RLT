//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Quantile
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Split a node
void Quan_Uni_Split_A_Node(size_t Node,
                          Reg_Uni_Tree_Class& OneTree,
                          const RLT_REG_DATA& REG_DATA,
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
      Reg_Uni_Terminate_Node(Node, OneTree, obs_id, REG_DATA.Y, REG_DATA.obsweight, useobsweight);

  }else{

    //Set up another split
    Split_Class OneSplit;

    //regular univariate split
    Quan_Uni_Find_A_Split(OneSplit,
                         REG_DATA,
                         Param,
                         (const uvec&) obs_id,
                         var_id,
                         rngl);
    
    // if did not find a good split, terminate
    if (OneSplit.score <= 0)
      goto TERMINATENODE;
    
    // record internal node mean 
    OneTree.NodeAve(Node) = arma::mean(REG_DATA.Y(obs_id));
    
    // construct indices for left and right nodes
    uvec left_id(obs_id.n_elem);
    
    if ( REG_DATA.Ncat(OneSplit.var) == 1 )
    {
      split_id(REG_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id); 
    }else{
      split_id_cat(REG_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id, REG_DATA.Ncat(OneSplit.var));
    }

    // if this happens something about the splitting rule is wrong
    if (left_id.n_elem == N or obs_id.n_elem == N)
      goto TERMINATENODE;
    
    // record internal node to tree 
    OneTree.SplitVar(Node) = OneSplit.var;
    OneTree.SplitValue(Node) = OneSplit.value;
    
    // check if the current tree is long enough to store two more nodes
    // if not, extend the current tree
    
    if ( OneTree.SplitVar( OneTree.SplitVar.n_elem - 2) != -2 )
    {
      if (Param.verbose)
      {
        RLTcout << "Tree extension needed. Terminal node size may not be well controlled." << std::endl;
      }
      
      OneTree.extend();
    }

    // get ready find the locations of next left and right nodes     
    size_t NextLeft = Node;
    size_t NextRight = Node;
    
    //Find locations of the next nodes
    OneTree.find_next_nodes(NextLeft, NextRight);
    
    OneTree.LeftNode(Node) = NextLeft;
    OneTree.RightNode(Node) = NextRight;

    // split the left and right nodes 
    Quan_Uni_Split_A_Node(NextLeft, 
                         OneTree,
                         REG_DATA,
                         Param,
                         left_id, 
                         var_id,
                         rngl);
    
    Quan_Uni_Split_A_Node(NextRight,                          
                         OneTree,
                         REG_DATA,
                         Param,
                         obs_id, 
                         var_id,
                         rngl);

  }
}

