//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Cla_Uni_Forest_Pred(cube& Pred,
                         const Cla_Uni_Forest_Class& CLA_FOREST,
                  			 const mat& X,
                  			 const uvec& Ncat,
                  			 size_t usecores,
                  			 size_t verbose)
{
  size_t N = X.n_rows;
  size_t ntrees = CLA_FOREST.SplitVarList.size();
  size_t nclass = CLA_FOREST.NodeProbList(0).n_cols;
  
  Pred.zeros(ntrees, nclass, N);
  
  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
        
      Tree_Class OneTree(CLA_FOREST.SplitVarList(nt),
                         CLA_FOREST.SplitValueList(nt),
                         CLA_FOREST.LeftNodeList(nt),
                         CLA_FOREST.RightNodeList(nt),
                         CLA_FOREST.NodeWeightList(nt));
      
      Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      for (size_t i = 0; i < N; i++)
      {
        Pred.slice(i).row(nt) = CLA_FOREST.NodeProbList(nt).row(TermNode(i));
      }
    }
    
    // #pragma omp barrier

  }
}