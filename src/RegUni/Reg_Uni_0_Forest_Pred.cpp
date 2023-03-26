//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Forest_Pred(mat& Pred,
                         const Reg_Uni_Forest_Class& REG_FOREST,
                  			 const mat& X,
                  			 const uvec& Ncat,
                  			 size_t usecores,
                  			 size_t verbose)
{
  size_t N = X.n_rows;
  size_t ntrees = REG_FOREST.SplitVarList.size();
  
  Pred.zeros(N, ntrees);
  
  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
        
      Reg_Uni_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                 REG_FOREST.SplitValueList(nt),
                                 REG_FOREST.LeftNodeList(nt),
                                 REG_FOREST.RightNodeList(nt),
                                 REG_FOREST.NodeWeightList(nt),
                                 REG_FOREST.NodeAveList(nt));
      
      Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      Pred.unsafe_col(nt).rows(real_id) = OneTree.NodeAve(TermNode);
    }
  }
}