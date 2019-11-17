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
              							const uvec& real_id,
              							uvec& TermNode)
{
 
 size_t size = proxy_id.n_elem;
  
  DEBUG_Rcout << "/// Start at node ///" << Node << " n is " << size << std::endl;
  
  if (OneTree.NodeType[Node] == 3)
  {
    for ( size_t i=0; i < size; i++ )
      TermNode(proxy_id(i)) = Node;
  }else{
    
    uvec id_goright(proxy_id.n_elem, fill::zeros);
    
    size_t SplitVar = OneTree.SplitVar(Node);
    double SplitValue = OneTree.SplitValue(Node);
    double xtemp = 0;
    
    if ( Ncat(SplitVar) > 1 ) // categorical var 
    {
      
      uvec goright(Ncat(SplitVar) + 1);
      unpack(SplitValue, Ncat(SplitVar) + 1, goright); // from Andy's rf package
      
      for (size_t i = 0; i < size ; i++)
      {
        xtemp = X( real_id( proxy_id(i) ), SplitVar);
        
        if ( goright( (size_t) xtemp ) == 1 )
          id_goright(i) = 1;
      }
      
    }else{
      
      for (size_t i = 0; i < size ; i++)
      {
        xtemp = X( real_id( proxy_id(i) ), SplitVar);
        
        if (xtemp > SplitValue)
          id_goright(i) = 1;
      }
    }
    
    uvec left_proxy = proxy_id(find(id_goright == 0));
    proxy_id = proxy_id(find(id_goright == 1));
    
    // left node 
    
    if (left_proxy.n_elem > 0)
    {
      Uni_Find_Terminal_Node(OneTree.LeftNode[Node], OneTree, X, Ncat, left_proxy, real_id, TermNode);
    }
    
    // right node
    if (proxy_id.n_elem > 0)
    {
      Uni_Find_Terminal_Node(OneTree.RightNode[Node], OneTree, X, Ncat, proxy_id, real_id, TermNode);      
    }
    
  }
  
  return;

}



void Uni_Find_Terminal_Node_ShuffleJ(size_t Node, 
                                    const Uni_Tree_Class& OneTree,
                                    const mat& X,
                                    const uvec& Ncat,
                                    uvec& proxy_id,
                                    const uvec& real_id,
                                    uvec& TermNode,
                                    const vec& tildex,
                                    const size_t j)
{
    
    size_t size = proxy_id.n_elem;
    
    DEBUG_Rcout << "/// Start at node ///" << Node << " n is " << size << std::endl;
    
    if (OneTree.NodeType[Node] == 3)
    {
        for ( size_t i=0; i < size; i++ )
            TermNode(proxy_id(i)) = Node;
    }else{
        
        uvec id_goright(proxy_id.n_elem, fill::zeros);
      
        size_t SplitVar = OneTree.SplitVar(Node);
        double SplitValue = OneTree.SplitValue(Node);
        double xtemp = 0;
        
        if ( Ncat(SplitVar) > 1 ) // categorical var 
        {

          uvec goright(Ncat(SplitVar) + 1);
          unpack(SplitValue, Ncat(SplitVar) + 1, goright); // from Andy's rf package
          
          //Rcout << "-- at Node " << Node << " go right is \n " << goright << std::endl;
          
          for (size_t i = 0; i < size ; i++)
          {
            if (SplitVar == j)
            {
              xtemp = tildex( proxy_id(i) );
              //Rcout << "-- original data is  " << X( real_id( proxy_id(i) ), SplitVar) << " permuted data is " << xtemp << std::endl;
              
            }else{
              xtemp = X( real_id( proxy_id(i) ), SplitVar);
            }

            
            
            if ( goright( (size_t) xtemp ) == 1 )
              id_goright(i) = 1;
          }

        }else{

          for (size_t i = 0; i < size ; i++)
          {
            if (SplitVar == j)
            {
              xtemp = tildex( proxy_id(i) );
            }else{
              xtemp = X( real_id( proxy_id(i) ), SplitVar);
            }
            
            if (xtemp > SplitValue)
              id_goright(i) = 1;
          }
        }
        
        uvec left_proxy = proxy_id(find(id_goright == 0));
        proxy_id = proxy_id(find(id_goright == 1));
        
        // left node 
        
        if (left_proxy.n_elem > 0)
        {
            Uni_Find_Terminal_Node_ShuffleJ(OneTree.LeftNode[Node], OneTree, X, Ncat, left_proxy, real_id, TermNode, tildex, j);
        }
        
        // right node
        if (proxy_id.n_elem > 0)
        {
            Uni_Find_Terminal_Node_ShuffleJ(OneTree.RightNode[Node], OneTree, X, Ncat, proxy_id, real_id, TermNode, tildex, j);      
        }
        
    }
    
    return;
    
}

