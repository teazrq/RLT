//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Utility/Utility.h"
# include "Trees.h"

using namespace Rcpp;
using namespace arma;

void Uni_Find_Terminal_Node(size_t Node, 
							const Uni_Tree_Class& OneTree,
							const mat& X,
							const uvec& Ncat,
							uvec& proxy_id,
							uvec& real_id,
							uvec& TermNode)
{
 
  size_t size = proxy_id.n_elem;
  
  DEBUG_Rcout << "/// Start at node ///" << Node << " n is " << size << std::endl;
   
  if (OneTree.NodeType[Node] == 3)
  {
    for ( size_t i=0; i < size; i++ )
      TermNode[proxy_id[i]] = Node;
  }else{
    
    uvec left_proxy(proxy_id.n_elem);
    size_t RightN = size;
    size_t LeftN = 0;
    size_t SplitVar = OneTree.SplitVar[Node];
    double SplitValue = OneTree.SplitValue[Node];    
    
    if ( Ncat(SplitVar) > 1 ) // categorical var 
    {
      size_t i = 0;
      
      uvec goright(Ncat[SplitVar] + 1);
      unpack(SplitValue, Ncat[SplitVar] + 1, goright); // from Andy's rf package
      
      while( i < RightN ){
//Rcout << " i is " << i << " left " << LeftN << " right " << RightN << std::endl;
        if ( goright(X(real_id[proxy_id[i]], SplitVar)) == 0 )
        {
          // move subject to left 
          left_proxy[LeftN++] = proxy_id[i];
          
          // remove subject from right 
          proxy_id[i] = proxy_id[RightN - 1];
          RightN--;
        }else{
          i++;
        }
        
        
      }
      
    }else{
      
      size_t i = 0;

      while( i < RightN ){
        //Rcout << " i is " << i << " left " << LeftN << " right " << RightN << std::endl;
        if ( X(real_id[proxy_id[i]], SplitVar) <= SplitValue )
        {
          // move subject to left 
          left_proxy[LeftN++] = proxy_id[i];
          
          // remove subject from right 
          proxy_id[i] = proxy_id[RightN - 1];
          RightN--;
        }else{
          i++;
        }
      }

    }
    
    // left node 
    
    if (LeftN > 0)
    {
      left_proxy.resize(LeftN);
      Uni_Find_Terminal_Node(OneTree.LeftNode[Node], OneTree, X, Ncat, left_proxy, real_id, TermNode);
    }
    
    // right node
    if (RightN > 0)
    {
      proxy_id.resize(RightN);
      Uni_Find_Terminal_Node(OneTree.RightNode[Node], OneTree, X, Ncat, proxy_id, real_id, TermNode);      
    }
    
  }
  
  return;

}