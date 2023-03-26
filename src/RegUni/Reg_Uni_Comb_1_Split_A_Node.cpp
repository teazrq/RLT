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
                               Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;
  size_t linear_comb = Param.linear_comb;
  
  if (Param.verbose)
    RLTcout << "at node " << Node << " ..." << std::endl;
  
  if (N <= nmin)
  {
TERMINATENODE:
    Reg_Uni_Comb_Terminate_Node(Node, OneTree, obs_id, REG_DATA.Y, REG_DATA.obsweight, useobsweight);
    
  }else{
    
    //Set up another split
    uvec var(linear_comb, fill::zeros);
    vec load(linear_comb, fill::zeros);
    
    Comb_Split_Class OneSplit(var, load);
    
    //Figure out where to split the node
    Reg_Uni_Comb_Find_A_Split(OneSplit, REG_DATA, Param, obs_id, var_id, rngl);
    
    RLTcout << "\n-back to Reg_Uni_Comb_Split_A_Node ... " << std::endl;
    OneSplit.print();
    
goto TERMINATENODE;
    
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
  
  OneTree.SplitVar(Node, 0) = -1; // -1 says this node is a terminal node. Ow, it would be the variable num
  
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
