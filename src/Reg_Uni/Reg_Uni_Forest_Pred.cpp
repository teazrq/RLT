//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../regForest.h"

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Forest_Pred(mat& Pred,
                         const Reg_Uni_Forest_Class& REG_FOREST,
                  			 const mat& X,
                  			 const uvec& Ncat,
                  			 const uvec& treeindex,
                  			 int usecores,
                  			 int verbose)
{
  
  size_t N = X.n_rows;
  size_t ntrees = REG_FOREST.NodeTypeList.size();
  
  Pred.zeros(N, treeindex.n_elem);

  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < treeindex.n_elem; nt++)
    {
      
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
      
      size_t whichtree = treeindex(nt);
        
      Reg_Uni_Tree_Class OneTree(REG_FOREST.NodeTypeList(whichtree), 
                                 REG_FOREST.SplitVarList(whichtree),
                                 REG_FOREST.SplitValueList(whichtree),
                                 REG_FOREST.LeftNodeList(whichtree),
                                 REG_FOREST.RightNodeList(whichtree),
                                 REG_FOREST.NodeSizeList(whichtree),
                                 REG_FOREST.NodeAveList(whichtree));
      
      Uni_Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      Pred.unsafe_col(nt).rows(real_id) = OneTree.NodeAve(TermNode);
    }
  }
}

