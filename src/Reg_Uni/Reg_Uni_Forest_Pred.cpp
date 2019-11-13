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
                         mat& W,
                         const Reg_Uni_Forest_Class& REG_FOREST,
                  			 const mat& X,
                  			 const uvec& Ncat,
                  			 bool kernel,
                  			 int usecores,
                  			 int verbose)
{
  
  size_t N = X.n_rows;
  size_t ntrees = REG_FOREST.NodeTypeList.size();
  
  
  Pred.set_size(N, ntrees);
  Pred.zeros();

  if (kernel)
  {
    W.set_size(N, ntrees);
    W.zeros();
  }

  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
      
      Reg_Uni_Tree_Class OneTree(REG_FOREST.NodeTypeList(nt), 
                                 REG_FOREST.SplitVarList(nt),
                                 REG_FOREST.SplitValueList(nt),
                                 REG_FOREST.LeftNodeList(nt),
                                 REG_FOREST.RightNodeList(nt),
                                 REG_FOREST.NodeSizeList(nt),
                                 REG_FOREST.NodeAveList(nt));
      
      Uni_Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      Pred.unsafe_col(nt).rows(real_id) = OneTree.NodeAve(TermNode);
      
      if (kernel)
          W.unsafe_col(nt).rows(real_id) = OneTree.NodeSize(TermNode);
    }
  }
}

