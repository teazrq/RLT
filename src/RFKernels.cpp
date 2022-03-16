//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Random Forest Kernel
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List Kernel_Self(arma::field<arma::ivec>& SplitVar,
                    arma::field<arma::vec>& SplitValue,
                    arma::field<arma::uvec>& LeftNode,
                    arma::field<arma::uvec>& RightNode,
                    arma::mat& X,
                    arma::uvec& Ncat,
                    size_t verbose)
{
  size_t N = X.n_rows;
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  umat K(N, N, fill::zeros);
  uvec real_id = linspace<uvec>(0, N-1, N);  
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Tree_Class OneTree(SplitVar(nt),
                           SplitValue(nt),
                           LeftNode(nt),
                           RightNode(nt));
    
    // initiate all observations
    uvec proxy_id = linspace<uvec>(0, N-1, N);
    uvec TermNode(N, fill::zeros);
    
    // get terminal node id
    Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
    
    //record
    uvec UniqueNode = unique(TermNode);
    
    for (auto j : UniqueNode)
    {
      uvec ID = real_id(find(TermNode == j));
      
      K.submat(ID, ID) += 1;
    }
  }
  
  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
}

// [[Rcpp::export()]]
List Kernel_Cross(arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::mat& X1,
                     arma::mat& X2,
                     arma::uvec& Ncat,
                     size_t verbose)
{
  size_t N1 = X1.n_rows;
  size_t N2 = X2.n_rows;
  
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  // each element for one testing subject 
  umat K(N1, N2, fill::zeros);
  
  uvec real_id1 = linspace<uvec>(0, N1-1, N1);
  uvec real_id2 = linspace<uvec>(0, N2-1, N2);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Tree_Class OneTree(SplitVar(nt),
                           SplitValue(nt),
                           LeftNode(nt),
                           RightNode(nt));
    
    // initiate all observations
    uvec proxy_id1 = linspace<uvec>(0, N1-1, N1);
    uvec proxy_id2 = linspace<uvec>(0, N2-1, N2);
    
    uvec TermNode1(N1, fill::zeros);
    uvec TermNode2(N2, fill::zeros);
    
    // get terminal node ids
    Find_Terminal_Node(0, OneTree, X1, Ncat, proxy_id1, real_id1, TermNode1);
    Find_Terminal_Node(0, OneTree, X2, Ncat, proxy_id2, real_id2, TermNode2);
    
    // record
    uvec UniqueNode = intersect(unique(TermNode1), unique(TermNode2));

    for (auto j : UniqueNode)
    {
      uvec ID1 = real_id1(find(TermNode1 == j));
      uvec ID2 = real_id2(find(TermNode2 == j));
      
      K.submat(ID1, ID2) += 1;
    }
  }

  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
  
}

// [[Rcpp::export()]]
List Kernel_Train(arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::mat& X1,
                     arma::mat& X2,
                     arma::uvec& Ncat,
                     arma::umat& ObsTrack,
                     size_t verbose)
{
  size_t N1 = X1.n_rows;
  size_t N2 = X2.n_rows;
  
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  // each element for one testing subject 
  umat K(N1, N2, fill::zeros);
  
  uvec real_id1 = linspace<uvec>(0, N1-1, N1);
  uvec real_id2 = linspace<uvec>(0, N2-1, N2);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Tree_Class OneTree(SplitVar(nt),
                           SplitValue(nt),
                           LeftNode(nt),
                           RightNode(nt));
    
    // initiate all observations
    uvec proxy_id1 = linspace<uvec>(0, N1-1, N1);
    uvec proxy_id2 = linspace<uvec>(0, N2-1, N2);
    
    uvec TermNode1(N1, fill::zeros);
    uvec TermNode2(N2, fill::zeros);
    
    // get terminal node ids
    Find_Terminal_Node(0, OneTree, X1, Ncat, proxy_id1, real_id1, TermNode1);
    Find_Terminal_Node(0, OneTree, X2, Ncat, proxy_id2, real_id2, TermNode2);
    
    // record
    uvec UniqueNode = intersect(unique(TermNode1), unique(TermNode2));
    uvec intreent = ObsTrack.col(nt);
    
    for (auto j : UniqueNode)
    {
      uvec ID1 = real_id1(find(TermNode1 == j));
      uvec ID2 = real_id2(find(TermNode2 == j && intreent > 0));
      
      K.submat(ID1, ID2) += 1;
    }
  }
  
  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
  
}