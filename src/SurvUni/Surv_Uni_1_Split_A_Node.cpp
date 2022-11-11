//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Split a node
void Surv_Uni_Split_A_Node(size_t Node,
                          Surv_Uni_Tree_Class& OneTree,
                          const RLT_SURV_DATA& SURV_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          const uvec& var_id,
                          Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;

  // in srf, it is N <= 2nmin
  // corrected to nmin by RZ
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

    //regular univariate split
    Surv_Uni_Find_A_Split(OneSplit,
                         SURV_DATA,
                         Param,
                         obs_id,
                         var_id,
                         rngl);

    // if did not find a good split, terminate
    if (OneSplit.score <= 0)
      goto TERMINATENODE;
    
    // record internal node mean 
    //OneTree.NodeAve(Node) = arma::mean(SURV_DATA.Y(obs_id));
    
    // construct indices for left and right nodes
    uvec left_id(obs_id.n_elem);
    
    if ( SURV_DATA.Ncat(OneSplit.var) == 1 )
    {
      split_id(SURV_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id); 
    }else{
      split_id_cat(SURV_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id, SURV_DATA.Ncat(OneSplit.var));
    }

    // if this happens something about the splitting rule is wrong
    if (left_id.n_elem == N or left_id.n_elem == 0)
      goto TERMINATENODE;
    
    // record internal node to tree 
    OneTree.SplitVar(Node) = OneSplit.var;
    OneTree.SplitValue(Node) = OneSplit.value;
    
    // check if the current tree is long enough to store two more nodes
    // if not, extend the current tree
    
    if ( OneTree.SplitVar( OneTree.SplitVar.n_elem - 2) != -2 )
    {
      RLTcout << "extension needed ..." << std::endl;
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
    Surv_Uni_Split_A_Node(NextLeft, 
                         OneTree,
                         SURV_DATA,
                         Param,
                         left_id, 
                         var_id,
                         rngl);
    
    Surv_Uni_Split_A_Node(NextRight,                          
                         OneTree,
                         SURV_DATA,
                         Param,
                         obs_id, 
                         var_id,
                         rngl);

  }
}

// terminate and record a node

void Surv_Uni_Terminate_Node(size_t Node,
                            Surv_Uni_Tree_Class& OneTree,
                            uvec& obs_id,
                            const uvec& Y,
                            const uvec& Censor,
                            const size_t NFail,
                            const vec& obs_weight,
                            bool useobsweight)
{

  OneTree.SplitVar(Node) = -1; // -1 says this node is a terminal node. Ow, it would be the variable num
  OneTree.LeftNode(Node) = obs_id.n_elem; // save node size on LeftNode
  
  //Find the average of the observations in the terminal node
  if (useobsweight)
  {
    //NOT IMPLEMENETED
    
  }else{

    vec NodeHazard(NFail + 1, fill::zeros);
    uvec NodeCensor(NFail + 1, fill::zeros);
    
    for (size_t i = 0; i < obs_id.n_elem; i++)
    {
      if (Censor(obs_id(i)) == 0)
        NodeCensor( Y(obs_id(i)) )++;
      else
        NodeHazard( Y(obs_id(i)) )++;
    }
    
    size_t N = obs_id.n_elem - NodeCensor(0);
    double h = 1;
    
    for (size_t j = 1; j < NFail + 1; j++)
    {
      if (N <= 0) break;
      
      h = NodeHazard(j) / N;
      N -= NodeHazard(j) + NodeCensor(j);
      NodeHazard(j) = h;
    }

    OneTree.NodeHaz(Node) = NodeHazard;

    // uvec NodeCensor(NFail + 1, fill::zeros);
    // for (size_t i = 0; i < obs_id.n_elem; i++)
    // {
    //   if (Censor(obs_id(i)) == 0)
    //     NodeCensor( Y(obs_id(i)) )++;
    //   else
    //     OneTree.NodeHaz(Node)( Y(obs_id(i)) )++;
    // }
    // 
    // size_t N = obs_id.n_elem - NodeCensor(0);
    // double h = 1;
    // 
    // for (size_t j = 1; j < NFail + 1; j++)
    // {
    //   if (N <= 0) break;
    //   
    //   h = OneTree.NodeHaz(Node)(j) / N;
    //   N -= OneTree.NodeHaz(Node)(j) + NodeCensor(j);
    //   OneTree.NodeHaz(Node)(j) = h;
    // }
  }
}
