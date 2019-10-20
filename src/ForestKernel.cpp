//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Kernel
//  **********************************

// my header file
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List ForestKernelUni(arma::field<arma::uvec>& NodeType,
          					 arma::field<arma::uvec>& SplitVar,
          					 arma::field<arma::vec>& SplitValue,
          					 arma::field<arma::uvec>& LeftNode,
          					 arma::field<arma::uvec>& RightNode,
          					 arma::field<arma::field<arma::uvec>>& NodeRegi,
          					 arma::imat& ObsTrack,
          					 arma::mat& X,
          					 arma::uvec& Ncat,
          					 arma::vec& obsweight,
          					 bool useobsweight,
          					 int usecores,
          					 int verbose)
{

  Rcout << "/// RLT Kernel Function ///" << std::endl;
  
  size_t Ntest = X.n_rows;
  size_t N = ObsTrack.n_rows;
  size_t ntrees = ObsTrack.n_cols; 
  
  // check number of cores
  usecores = checkCores(usecores, verbose);
  
  // initiate output kernel as a field
  // each element for one testing subject 
  
  field<arma::mat> Kernel(Ntest);
  arma::mat blank(N, ntrees, fill::zeros);
  Kernel.fill(blank);

  #pragma omp parallel num_threads(usecores)
  {
  #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      DEBUG_Rcout << "--- on tree " << nt << std::endl;
      
      Uni_Tree_Class OneTree;
      OneTree.readin(NodeType[nt], SplitVar[nt], SplitValue[nt], LeftNode[nt], RightNode[nt]);

      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, Ntest-1, Ntest);
      uvec real_id = linspace<uvec>(0, Ntest-1, Ntest);
      uvec TermNode(Ntest, fill::zeros);
      
      Uni_Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      DEBUG_Rcout << "--- ready to record each subject " << std::endl;
      
      for (size_t i = 0; i < Ntest; i++)
      {
        size_t TreeNode = TermNode(i);
        uvec neighbers = NodeRegi[nt][TreeNode];
        
        if (useobsweight)
        {
          Kernel[i].unsafe_col(nt).rows(neighbers) += obsweight(neighbers);
        }else{
          vec one(neighbers.n_elem, fill::ones);
          Kernel[i].unsafe_col(nt).rows(neighbers) += one;
        }
      }
      
      DEBUG_Rcout << "--- finishing record each subject " << std::endl;
    }
  }

  List ReturnList;
  ReturnList["Kernel"] = Kernel;
  
  return(ReturnList);
  
}
