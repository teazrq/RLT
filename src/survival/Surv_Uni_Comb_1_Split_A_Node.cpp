//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

//Split a node (combination split)
void Surv_Uni_Comb_Split_A_Node(size_t Node,
                                Surv_Uni_Comb_Tree_Class& OneTree,
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
  size_t linear_comb = Param.linear_comb;

  if (N <= nmin)
  {
    TERMINATENODE:
    Surv_Uni_Comb_Terminate_Node(Node, OneTree, obs_id,
                                  SURV_DATA.Y, SURV_DATA.Censor,
                                  SURV_DATA.NFail,
                                  SURV_DATA.obsweight, useobsweight);

  }else{

    //Set up another split
    uvec var(linear_comb, fill::zeros);
    vec load(linear_comb, fill::zeros);
    Comb_Split_Class OneSplit(var, load);

    // update protected variables
    uvec new_var_id(var_id);
    uvec new_var_protect(var_protect);

    //Figure out where to split the node
    Surv_Uni_Comb_Find_A_Split(OneSplit, SURV_DATA, Param, obs_id,
                                new_var_id, new_var_protect, rngl);

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
    size_t n_comb = sum(OneSplit.load != 0);

    if ( n_comb == 1 ) // single variable split
    {
      SURV_DATA.Ncat(OneSplit.var(0)) == 1 ?
            split_id(SURV_DATA.X.unsafe_col(OneSplit.var(0)),
                     OneSplit.value,
                     left_id,
                     obs_id) :
            split_id_cat(SURV_DATA.X.unsafe_col(OneSplit.var(0)),
                         OneSplit.value,
                         left_id,
                         obs_id,
                         SURV_DATA.Ncat(OneSplit.var(0)));
    }else{ // combination split
      split_id_comb(SURV_DATA.X,
                    OneSplit,
                    left_id,
                    obs_id);
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
    OneTree.SplitVar.row(Node) = conv_to<irowvec>::from(OneSplit.var);
    OneTree.SplitLoad.row(Node) = conv_to<rowvec>::from(OneSplit.load);
    OneTree.SplitValue(Node) = OneSplit.value;

    // check if the current tree is long enough to store two more nodes
    if ( OneTree.SplitVar(OneTree.SplitVar.n_rows - 2, 0) != NodeStatus::Unused )
    {
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
    Surv_Uni_Comb_Split_A_Node(NextLeft,
                                OneTree,
                                SURV_DATA,
                                Param,
                                left_id,
                                new_var_id,
                                new_var_protect,
                                rngl);

    Surv_Uni_Comb_Split_A_Node(NextRight,
                                OneTree,
                                SURV_DATA,
                                Param,
                                obs_id,
                                new_var_id,
                                new_var_protect,
                                rngl);

  }
}

// terminate and record a node (combination split)

void Surv_Uni_Comb_Terminate_Node(size_t Node,
                                  Surv_Uni_Comb_Tree_Class& OneTree,
                                  uvec& obs_id,
                                  const uvec& Y,
                                  const uvec& Censor,
                                  const size_t NFail,
                                  const vec& obs_weight,
                                  bool useobsweight)
{
  OneTree.SplitVar(Node, 0) = -1; // -1: terminal node

  if (useobsweight)
  {
    // Weighted Nelson-Aalen CHF using normalized weights
    vec NodeHazard(NFail + 1, fill::zeros);
    vec NodeCensor_w(NFail + 1, fill::zeros);
    
    // Normalize weights: omega_i = w_i / mean(w)
    double wsum = 0;
    for (size_t i = 0; i < obs_id.n_elem; i++)
      wsum += obs_weight(obs_id(i));
    double wbar = wsum / obs_id.n_elem;
    
    for (size_t i = 0; i < obs_id.n_elem; i++)
    {
      double wi = survival_node_weight(obs_weight(obs_id(i)), wsum, wbar, obs_id.n_elem);
      if (Censor(obs_id(i)) == 0)
        NodeCensor_w( Y(obs_id(i)) ) += wi;
      else
        NodeHazard( Y(obs_id(i)) ) += wi;
    }
    
    // Weighted at-risk count
    double N_w = 0;
    for (size_t j = 0; j <= NFail; j++)
      N_w += NodeHazard(j) + NodeCensor_w(j);
    N_w -= NodeCensor_w(0);
    
    double h = 1;
    for (size_t j = 1; j < NFail + 1; j++)
    {
      if (N_w <= 0) break;
      
      h = NodeHazard(j) / N_w;
      N_w -= NodeHazard(j) + NodeCensor_w(j);
      NodeHazard(j) = h;
    }
    
    OneTree.NodeHaz(Node) = NodeHazard;
    OneTree.NodeWeight(Node) = wsum;
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
    OneTree.NodeWeight(Node) = obs_id.n_elem;
  }
}
