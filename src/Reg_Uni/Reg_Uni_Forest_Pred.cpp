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

vec Reg_Uni_Forest_Pred(const std::vector<Reg_Uni_Tree_Class>& Forest,
            						const mat& X,
            						const uvec& Ncat,
            						bool kernel,
            						int usecores,
            						int verbose)
{
  DEBUG_Rcout << "/// Start prediction ///" << std::endl;
  
  size_t N = X.n_rows;
  size_t ntrees = Forest.size();
    
  mat A(N, ntrees, fill::zeros);
  
  mat W;
  
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
      
      if (kernel)
      {
        A.unsafe_col(nt).rows(real_id) = Forest[nt].NodeAve(TermNode) % Forest[nt].NodeSize(TermNode);
        W.unsafe_col(nt).rows(real_id) = Forest[nt].NodeSize(TermNode);
      }else{
        A.unsafe_col(nt).rows(real_id) = Forest[nt].NodeAve(TermNode);
      }
    }
  }
  
  if (kernel)
  {
    return(sum(A, 1) / sum(W, 1));
  }else{
    return(mean(A, 1)); 
  }

}

