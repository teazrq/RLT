//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

//Split a node (embed/reinforcement mode)
void Surv_Uni_Split_A_Node_Embed(size_t Node,
                                 Surv_Uni_Tree_Class& OneTree,
                                 const RLT_SURV_DATA& SURV_DATA,
                                 const PARAM_GLOBAL& Param,
                                 uvec& obs_id,
                                 const uvec& var_id,
                                 const uvec& var_protect,
                                 Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;

  // in rf, it is N <= nmin
  if (N <= nmin)
  {
    TERMINATENODE:
    Surv_Uni_Terminate_Node(Node, OneTree, obs_id, 
                            SURV_DATA.Y, SURV_DATA.Censor,
                            SURV_DATA.NFail,
                            SURV_DATA.obsweight, useobsweight);

  }else{

    //Set up another split
    Split_Class OneSplit;
    
    uvec new_var_id(var_id);
    uvec new_var_protect(var_protect);
    
    Surv_Uni_Find_A_Split_Embed(OneSplit,
                                SURV_DATA,
                                Param,
                                (const uvec&) obs_id,
                                new_var_id,
                                new_var_protect,
                                rngl);

    // if did not find a good split, terminate
    if (OneSplit.score <= 0)
      goto TERMINATENODE;
    
    // record internal node weight 
    if (useobsweight)
    {
      OneTree.NodeWeight(Node) = arma::sum(SURV_DATA.obsweight(obs_id));
    }else{
      OneTree.NodeWeight(Node) = obs_id.n_elem;
    }
      
    // construct indices for left and right nodes
    uvec left_id(obs_id.n_elem);
    
    if ( SURV_DATA.Ncat(OneSplit.var) == 1 )
    {
      split_id(SURV_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id); 
    }else{
      split_id_cat(SURV_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id, SURV_DATA.Ncat(OneSplit.var));
    }

    // if this happens something about the splitting rule is wrong
    if (left_id.n_elem == 0 || left_id.n_elem == N ||
        (useobsweight && !positive_child_weights(SURV_DATA.obsweight, left_id, obs_id)))
    {
      // A terminal estimate must use the complete positive-mass parent.
      obs_id = join_cols(left_id, obs_id);
      goto TERMINATENODE;
    }
    
    // record internal node to tree 
    OneTree.SplitVar(Node) = OneSplit.var;
    OneTree.SplitValue(Node) = OneSplit.value;
    
    // check if the current tree is long enough to store two more nodes
    // if not, extend the current tree
    
    if ( OneTree.SplitVar( OneTree.SplitVar.n_elem - 2) != NodeStatus::Unused )
    {
      if (Param.verbose)
        RLTcout << "Tree extension needed. Terminal node size may not be well controlled." << std::endl;
      
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
    Surv_Uni_Split_A_Node_Embed(NextLeft, 
                                 OneTree,
                                 SURV_DATA,
                                 Param,
                                 left_id, 
                                 new_var_id,
                                 new_var_protect,
                                 rngl);
  
    
    Surv_Uni_Split_A_Node_Embed(NextRight,                          
                                 OneTree,
                                 SURV_DATA,
                                 Param,
                                 obs_id, 
                                 new_var_id,
                                 new_var_protect,
                                 rngl);
  }
}
