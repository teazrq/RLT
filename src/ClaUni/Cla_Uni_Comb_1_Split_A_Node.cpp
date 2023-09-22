//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Split a node
void Cla_Uni_Comb_Split_A_Node(size_t Node,
                               Cla_Uni_Comb_Tree_Class& OneTree,
                               const RLT_CLA_DATA& CLA_DATA,
                               const PARAM_GLOBAL& Param,
                               uvec& obs_id,
                               const uvec& var_id,
                               const uvec& var_protect,
                               Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;
  size_t linear_comb = Param.linear_comb;
  
  Cla_Uni_Record_Node(Node, OneTree.NodeWeight, OneTree.NodeProb, 
                      obs_id, 
                      CLA_DATA.Y, CLA_DATA.nclass, CLA_DATA.obsweight, 
                      useobsweight);

  // in rf, it is N <= nmin
  if ( N <= nmin or OneTree.NodeProb.row(Node).max() == 1 )
  {
TERMINATENODE:
    OneTree.SplitVar(Node, 0) = -1;
    return;
    
  }else{
    //Set up another split
    uvec var(linear_comb, fill::zeros);
    vec load(linear_comb, fill::zeros);
    Comb_Split_Class OneSplit(var, load);
    
    // update protected variables
    uvec new_var_id(var_id);
    uvec new_var_protect(var_protect);    
    
    //regular univariate split
    Cla_Uni_Comb_Find_A_Split(OneSplit,
                              CLA_DATA,
                              Param,
                              obs_id,
                              new_var_id,
                              new_var_protect,
                              rngl);
    
    // if did not find a good split, terminate
    if (OneSplit.score <= 0)
      goto TERMINATENODE;
    
    // construct indices for left and right nodes
    uvec left_id(obs_id.n_elem);
    
    size_t n_comb = sum(OneSplit.load != 0);    
    
    if ( n_comb == 1 ) // single variable split
    {
      CLA_DATA.Ncat(OneSplit.var(0)) == 1 ? 
      split_id(CLA_DATA.X.unsafe_col(OneSplit.var(0)), 
               OneSplit.value, 
               left_id, 
               obs_id) : 
      split_id_cat(CLA_DATA.X.unsafe_col(OneSplit.var(0)), 
                   OneSplit.value, 
                   left_id, 
                   obs_id, 
                   CLA_DATA.Ncat(OneSplit.var(0)));
    }else{ // combination split
      split_id_comb(CLA_DATA.X,
                    OneSplit,
                    left_id,
                    obs_id);
    }
    
    // if this happens something about the splitting rule is wrong
    if (left_id.n_elem == N or obs_id.n_elem == N)
      goto TERMINATENODE;
    
    // record internal node to tree 
    OneTree.SplitVar.row(Node) = conv_to<irowvec>::from(OneSplit.var);
    OneTree.SplitLoad.row(Node) = conv_to<rowvec>::from(OneSplit.load);
    OneTree.SplitValue(Node) = OneSplit.value;
    
    // check if the current tree is long enough to store two more nodes
    // if not, extend the current tree
    if ( OneTree.SplitVar( OneTree.SplitVar.n_elem - 2) != -2 )
    {
      if (Param.verbose)
        RLTcout << "Tree extension needed. Report Error." << std::endl;
      
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
    Cla_Uni_Comb_Split_A_Node(NextLeft, 
                              OneTree,
                              CLA_DATA,
                              Param,
                              left_id, 
                              new_var_id,
                              new_var_protect,
                              rngl);
    
    Cla_Uni_Comb_Split_A_Node(NextRight,                          
                              OneTree,
                              CLA_DATA,
                              Param,
                              obs_id, 
                              new_var_id,
                              new_var_protect,
                              rngl);
  }
}
