//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"

using namespace Rcpp;
using namespace arma;

mat Surv_Uni_Forest_Pred(const std::vector<Surv_Uni_Tree_Class>& Forest,
            						const mat& X,
            						const uvec& Ncat,
            						int NFail,
            						bool kernel,
            						int usecores,
            						int verbose)
{
  DEBUG_Rcout << "/// Start prediction ///" << std::endl;
  
  size_t N = X.n_rows;
  size_t ntrees = Forest.size();

  cube A(NFail + 1, ntrees, N, fill::zeros);
  
  mat W(N, ntrees, fill::zeros);
  
  if (kernel)
  {
    W.set_size(N, ntrees);
    W.zeros();
  }
  
  mat Pred(N, NFail + 1);
    
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
        //A.unsafe_col(nt).rows(real_id) = Forest[nt].NodeAve(TermNode) % Forest[nt].NodeSize(TermNode);
        //W.unsafe_col(nt).rows(real_id) = Forest[nt].NodeSize(TermNode);
      }else{
        for (size_t i = 0; i < N; i++)
        {
          A.slice(i).col(nt) = Forest[nt].NodeHaz[TermNode(i)];
        }
      }
    }
    
    #pragma omp barrier
    for (size_t i = 0; i < N; i++)
    {
      Pred.row(i) = mean(A.slice(i), 1).t();
    }
  }
  
  Pred.shed_col(0);

  DEBUG_Rcout << " sub 1 \n" << Pred << std::endl;
  
  return(Pred);
  /*
  if (kernel)
  {
    return(sum(A, 1) / sum(W, 1));
  }else{
    return(mean(A, 1)); 
  }
  */
}

