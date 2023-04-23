//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Split a node
void Reg_Uni_Comb_Split_A_Node(size_t Node,
                               Reg_Uni_Comb_Tree_Class& OneTree,
                               const RLT_REG_DATA& REG_DATA,
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
  
  if (N <= nmin)
  {
TERMINATENODE:
    Reg_Uni_Comb_Terminate_Node(Node, OneTree, obs_id, REG_DATA.Y, REG_DATA.obsweight, useobsweight);
    
  }else{
    
    //Set up another split
    uvec var(linear_comb, fill::zeros);
    vec load(linear_comb, fill::zeros);
    Comb_Split_Class OneSplit(var, load);
    
    // update protected variables
    uvec new_var_id(var_id);
    uvec new_var_protect(var_protect);
    
    //Figure out where to split the node
    Reg_Uni_Comb_Find_A_Split(OneSplit, REG_DATA, Param, obs_id, 
                              new_var_id, new_var_protect, rngl);
    
    // if did not find a good split, terminate
    if (OneSplit.score <= 0)
      goto TERMINATENODE;
      
    // record internal node weight 
    if (useobsweight)
    {
      OneTree.NodeWeight(Node) = arma::sum(REG_DATA.obsweight(obs_id));
    }else{
      OneTree.NodeWeight(Node) = obs_id.n_elem;
    }    
    
    // construct indices for left and right nodes
    uvec left_id(obs_id.n_elem);
    size_t n_comb = sum(OneSplit.load != 0);

    // RLTcout << " n_comb is " << n_comb << " at node " << Node << std::endl;
    
    if ( n_comb == 1 ) // single variable split
    {
      REG_DATA.Ncat(OneSplit.var(0)) == 1 ? 
            split_id(REG_DATA.X.unsafe_col(OneSplit.var(0)), 
                     OneSplit.value, 
                     left_id, 
                     obs_id) : 
            split_id_cat(REG_DATA.X.unsafe_col(OneSplit.var(0)), 
                         OneSplit.value, 
                         left_id, 
                         obs_id, 
                         REG_DATA.Ncat(OneSplit.var(0)));
    }else{ // combination split
      split_id_comb(REG_DATA.X,
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
    
    if ( OneTree.SplitVar(OneTree.SplitVar.n_rows - 2, 0) != -2 )
    {
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
    Reg_Uni_Comb_Split_A_Node(NextLeft,
                              OneTree,
                              REG_DATA,
                              Param,
                              left_id,
                              new_var_id,
                              new_var_protect,
                              rngl);
    
    Reg_Uni_Comb_Split_A_Node(NextRight,
                              OneTree,
                              REG_DATA,
                              Param,
                              obs_id,
                              new_var_id,
                              new_var_protect,
                              rngl);    

  }
}

// terminate and record a node

void Reg_Uni_Comb_Terminate_Node(size_t Node,
              								   Reg_Uni_Comb_Tree_Class& OneTree,
              								   uvec& obs_id,
              								   const vec& Y,
              								   const vec& obs_weight,
              								   bool useobsweight)
{
  // -1: terminal node
  OneTree.SplitVar(Node, 0) = -1; 
  
  //Find the average of the observations in the terminal node
  if (useobsweight)
  {
    double allweight = arma::sum(obs_weight(obs_id));
    OneTree.NodeWeight(Node) = allweight; // save total weights
    OneTree.NodeAve(Node) = arma::sum(Y(obs_id) % obs_weight(obs_id)) / allweight;
  }else{
    OneTree.NodeWeight(Node) = obs_id.n_elem; // save node weight
    OneTree.NodeAve(Node) = arma::mean(Y(obs_id));
  }
}
