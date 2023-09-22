//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
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
  
  Cla_Uni_Record_Node(Node, OneTree.NodeWeight, OneTree.NodeProb,
                      obs_id, 
                      CLA_DATA.Y, CLA_DATA.nclass, CLA_DATA.obsweight, 
                      useobsweight);

  // in rf, it is N <= nmin
  if (N <= nmin or OneTree.NodeProb.row(Node).max() == 1 )
  {
TERMINATENODE:
    OneTree.SplitVar(Node) = -1;
    return;
    
  }else{
    //Set up another split
    Split_Class OneSplit;
    
    //regular univariate split
    Cla_Uni_Find_A_Split(OneSplit,
                         CLA_DATA,
                         Param,
                         (const uvec&) obs_id,
                         var_id,
                         rngl);
    
    // if did not find a good split, terminate
    if (OneSplit.score <= 0)
      goto TERMINATENODE;
    
    // construct indices for left and right nodes
    uvec left_id(obs_id.n_elem);
    
    if ( CLA_DATA.Ncat(OneSplit.var) == 1 )
    {
      split_id(CLA_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id); 
    }else{
      split_id_cat(CLA_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id, CLA_DATA.Ncat(OneSplit.var));
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
    Cla_Uni_Split_A_Node(NextLeft, 
                         OneTree,
                         CLA_DATA,
                         Param,
                         left_id, 
                         var_id,
                         rngl);
    
    Cla_Uni_Split_A_Node(NextRight,                          
                         OneTree,
                         CLA_DATA,
                         Param,
                         obs_id, 
                         var_id,
                         rngl);
  }
}

// terminate and record a node

void Cla_Uni_Record_Node(size_t Node,
                         arma::vec& TreeNodeWeight,
                         arma::mat& TreeNodeProb,
                         uvec& obs_id,
                         const uvec& Y,
                         const size_t nclass,
                         const vec& obs_weight,
                         bool useobsweight)
{
  // calculate node probability
  if (useobsweight)
  {
    double allweight = arma::sum(obs_weight(obs_id));
    vec nodecount(nclass, fill::zeros);
    
    // get node prob
    uvec labels = Y(obs_id);
    vec weights = obs_weight(obs_id);
    nodecount(labels) += weights;
    
    // save node weight
    TreeNodeWeight(Node) = allweight;
    TreeNodeProb.row(Node) = nodecount / allweight;
  }else{
    
    rowvec nodecount(nclass, fill::zeros);
    uvec labels = Y(obs_id);
    nodecount(labels) += 1;
    
    // save node count
    TreeNodeWeight(Node) = obs_id.n_elem;
    TreeNodeProb.row(Node) = nodecount / obs_id.n_elem;
  }
}
