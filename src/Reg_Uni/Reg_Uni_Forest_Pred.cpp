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
                         const std::vector<Reg_Uni_Tree_Class>& Forest,
            			 const mat& X,
            			 const uvec& Ncat,
            			 bool kernel,
            			 int usecores,
            			 int verbose)
{
  DEBUG_Rcout << "/// Start prediction ///" << std::endl;
  
  size_t N = X.n_rows;
  size_t ntrees = Forest.size();
    
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
      
      Uni_Find_Terminal_Node(0, Forest[nt], X, Ncat, proxy_id, real_id, TermNode);
      
      Pred.unsafe_col(nt).rows(real_id) = Forest[nt].NodeAve(TermNode);
      
      if (kernel)
          W.unsafe_col(nt).rows(real_id) = Forest[nt].NodeSize(TermNode);
    }
  }
}

