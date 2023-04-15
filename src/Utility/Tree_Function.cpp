//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Tree arranging functions
//  **********************************

// my header file
# include "Tree_Definition.h"
# include "Tree_Function.h"

using namespace Rcpp;
using namespace arma;

// categorical variable packing
// some translated from Andy's rf package
double pack(const size_t nBits, const uvec& bits)
{
  double value = bits(nBits - 1);

  for (int i = nBits - 2; i >= 0; i--)
    value = 2.0*value + bits(i);
  
  return(value);
}

void unpack(const double pack, const size_t nBits, uvec& bits)
{
  double x = pack;
  for (size_t i = 0; i < nBits; ++i)
  {
    bits(i) = ((size_t) x & 1) ? 1 : 0;
    x /= 2;
  }
}

bool unpack_goright(double pack, const size_t cat)
{
  for (size_t i = 0; i < cat; i++) pack /= 2;
  return(((size_t) pack & 1) ? 1 : 0);
}

void goright_roll_one(arma::uvec& goright_cat)
{
  size_t n = goright_cat.n_elem;
  goright_cat(0) ++;
  
  for (size_t i = 0; i < n-1; i ++)
  {
    if (goright_cat(i) == 2)
    {
      goright_cat(i) = 0;
      goright_cat(i+1)++;
    }
  }
  
  if (goright_cat(n-1) > 1)
    RLTcout << "goright_cat reaches max" << std::endl;
}

// for resampling set ObsTrack
void set_obstrack(arma::imat& ObsTrack,
                  const size_t nt,
                  const size_t size,
                  const bool replacement,
                  Rand& rngl)
{
  
  size_t N = ObsTrack.n_rows;
  arma::uvec insample;
  
  insample = rngl.sample(0, N-1, size, replacement);

  for (size_t i = 0; i < size; i++)
    ObsTrack(insample(i), nt) ++;
	
}

// get inbag and oobag samples from ObsTrack
// Adjusted for new ObsTrack format
void get_samples(arma::uvec& inbagObs,
                 arma::uvec& oobagObs,
                 const arma::uvec& subj_id,
                 const arma::ivec& ObsTrack_nt)
{
  //oob samples
	oobagObs = subj_id.elem( find(ObsTrack_nt == 0) );
  
  //inbag samples
  arma::uvec use_row = find(ObsTrack_nt > 0);
	size_t N = sum( ObsTrack_nt.elem( use_row ) );
	inbagObs.set_size(N);
	
	// record those to inbagObs
	size_t mover = 0;
	for (auto i : use_row)
		for (int k = 0; k < ObsTrack_nt(i); k++)
			inbagObs(mover++) = subj_id(i);
}


// splitting an interval node
// construct id vectors for left and right nodes
void split_id(const vec& x, double value, uvec& left_id, uvec& obs_id) // obs_id will be treated as the right node
{
  size_t LeftN = 0;
  size_t RightN = 0;
  
  for (size_t i = 0; i < obs_id.n_elem; i++)
  {
    if ( x(obs_id(i)) <= value )
      left_id(LeftN++) = obs_id(i);
    else
      obs_id(RightN++) = obs_id(i);
  }
  
  left_id.resize(LeftN);
  obs_id.resize(RightN);
}

void split_id_cat(const vec& x, double value, uvec& left_id, uvec& obs_id, size_t ncat) // obs_id will be treated as the right node
{
  // the first (0-th) element (category) of goright will always be set to 0 --- go left, 
  // but this category does not exist.
  uvec goright(ncat + 1, fill::zeros);   
  unpack(value, ncat + 1, goright);
  
  size_t LeftN = 0;
  size_t RightN = 0;
  
  for (size_t i = 0; i < obs_id.n_elem; i++)
  {
    if ( goright[x(obs_id(i))] == 0 )
      left_id(LeftN++) = obs_id(i);
    else
      obs_id(RightN++) = obs_id(i);
  }
  
  left_id.resize(LeftN);
  obs_id.resize(RightN);
}

// check cutoff points in continuous or categorical variables
void check_cont_index_sub(size_t& lowindex, 
                          size_t& highindex, 
                          const vec& x,
                          const uvec& indices)
{
  // x(indices) must be sorted
  // ties must already happened when running this function 
  // also x must contain different elements
  size_t N = indices.n_elem;
  
  // as long as x at lowindex is not the same as minimum, push it lower to a non-tie
  if ( x(indices(lowindex)) > x(indices(0)) )
    while ( x(indices(lowindex)) == x(indices(lowindex+1)) ) lowindex--;
  else // otherwise, move up
    while ( x(indices(lowindex)) == x(indices(lowindex+1)) ) lowindex++;
  
  // as long as x at highindex does not the same as maximum, push it higher to a non-tie
  if ( x(indices(highindex)) < x(indices(N-1)) )
      while ( x(indices(highindex)) == x(indices(highindex+1)) ) highindex++;
  else // otherwise move down
      while ( x(indices(highindex)) == x(indices(highindex+1)) ) highindex--;
}


void check_cont_index(size_t& lowindex, 
                      size_t& highindex, 
                      const vec& x)
{
  // x must be sorted
  // ties must already happened when running this function 
  // also x must contain different elements
  
  // check with extremes
  // make sure lowindex has space to move up
  while (x(lowindex) == x(x.n_elem-1)) lowindex--;
  
  // make sure highindex has space to move down
  while (x(highindex+1) == x(0)) highindex++;
  
  // make sure lowindex is not at a tie, o.w. move up
  while (x(lowindex) == x(lowindex+1)) lowindex++;
  
  // make sure highindex is not at a tie, o.w. move down
  while (x(highindex) == x(highindex+1)) highindex--;  

}


// for categorical variables
void move_cat_index(size_t& lowindex, 
                    size_t& highindex, 
                    std::vector<Cat_Class*>& cat_reduced, 
                    size_t true_cat, 
                    size_t nmin)
{
  lowindex = 0;
  highindex = true_cat - 2;
  
  if (true_cat == 2) //nothing we can do
    return; 
  
  size_t lowcount = cat_reduced[0]->count;
  size_t highcount = cat_reduced[true_cat-1]->count;
  
  // now both low and high index are not tied with the end
  if ( lowcount >= nmin and highcount >= nmin ) // everything is good
    return;
  
  if ( lowcount < nmin and highcount >= nmin ) // only need to fix lowindex
  {
    while( lowcount < nmin and lowindex <= highindex ){
      lowindex++;
      lowcount += cat_reduced[lowindex]->count;
    }
    
    if ( lowindex > highindex ) lowindex = highindex;
    
    return;
  }
  
  if ( lowcount >= nmin and highcount < nmin ) // only need to fix highindex
  {
    while( highcount < nmin and lowindex <= highindex ){
      highcount += cat_reduced[highindex]->count;
      highindex--;
    }
    
    // sometimes highindex will be negative and turned into very large number 
    if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex; 
    return;
  }
  
  // if both need to be fixed, start with one randomly
  if ( lowcount < nmin and highcount < nmin ) 
  {
    if ( TRUE ) // we can fix this later with random choice
    { // fix lowindex first
      while( lowcount < nmin and lowindex <= highindex ){
        lowindex++;
        lowcount += cat_reduced[lowindex]->count;
      }
      
      if (lowindex > highindex ) lowindex = highindex;
      
      while( highcount < nmin and lowindex <= highindex ){
        highcount += cat_reduced[highindex]->count;
        highindex--;
      }
      
      if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex;
      
      return;
      
    }else{ // fix highindex first
      while( highcount < nmin and lowindex <= highindex ){
        highcount += cat_reduced[highindex]->count;
        highindex--;
      }
      
      if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex;
      
      while( lowcount < nmin and lowindex <= highindex ){
        lowindex++;
        lowcount += cat_reduced[lowindex]->count;
      }
      
      if (lowindex > highindex) lowindex = highindex;
      
      return;
    }
  }
}

// for sorting a list of categorical class 
bool cat_class_compare(Cat_Class& a, Cat_Class& b)
{
  if (a.count == 0 and b.count == 0)
    return 0;
  
  if (a.count > 0 and b.count == 0)
    return 1;
  
  if (a.count == 0 and b.count > 0)
    return 0;
  
  return ( a.score < b.score );
}

// Find the terminal node for X in one tree
void Find_Terminal_Node(size_t Node, 
                            const Tree_Class& OneTree,
                            const mat& X,
                            const uvec& Ncat,
                            uvec& proxy_id,
                            const uvec& real_id,
                            uvec& TermNode)
{
  
  size_t size = proxy_id.n_elem;
  
  //If the current node is a terminal node
  if (OneTree.SplitVar[Node] == -1)
  {
    // For all the observations in the node,
    // Set its terminal node
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
      
      //For the obs in the current internal node
      for (size_t i = 0; i < size ; i++)
      {
        //Determine the x values for this variable
        xtemp = X( real_id( proxy_id(i) ), SplitVar);
        
        //If they are greater than the value, go right
        if (xtemp > SplitValue)
          id_goright(i) = 1;
      }
    }
    
    //All others go left
    uvec left_proxy = proxy_id(find(id_goright == 0));
    proxy_id = proxy_id(find(id_goright == 1));
    
    // left node 
    
    if (left_proxy.n_elem > 0)
    {
      Find_Terminal_Node(OneTree.LeftNode[Node], OneTree, X, Ncat, left_proxy, real_id, TermNode);
    }
    
    // right node
    if (proxy_id.n_elem > 0)
    {
      Find_Terminal_Node(OneTree.RightNode[Node], OneTree, X, Ncat, proxy_id, real_id, TermNode);      
    }
    
  }
  
  return;
  
}

// Find the terminal node for X in one tree with variable j shuffled
//Function for variable importance
void Find_Terminal_Node_ShuffleJ(size_t Node, 
                                     const Tree_Class& OneTree,
                                     const mat& X,
                                     const uvec& Ncat,
                                     uvec& proxy_id,
                                     const uvec& real_id,
                                     uvec& TermNode,
                                     const vec& tildex,
                                     const size_t j)
{
  
  size_t size = proxy_id.n_elem;
  
  //If terminal node
  if (OneTree.SplitVar[Node] == -1)
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
        if (SplitVar == j)
        {
          xtemp = tildex( proxy_id(i) );
          
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
          // If it is the shuffle variable, randomly get x
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
      Find_Terminal_Node_ShuffleJ(OneTree.LeftNode[Node], OneTree, X, Ncat, left_proxy, real_id, TermNode, tildex, j);
    }
    
    // right node
    if (proxy_id.n_elem > 0)
    {
      Find_Terminal_Node_ShuffleJ(OneTree.RightNode[Node], OneTree, X, Ncat, proxy_id, real_id, TermNode, tildex, j);      
    }
    
  }
  
  return;
  
}

// find terminal weight given the randomness of one variable 
void Assign_Terminal_Node_Prob_RandomJ(size_t Node,
                                       const Tree_Class& OneTree,
                                       const mat& X,
                                       const uvec& Ncat,
                                       size_t id,
                                       double Multipler,
                                       vec& Prob,
                                       size_t j)
{
  // If the current node is a terminal node
  if ( OneTree.SplitVar[Node] == -1 )
  {
    // Assign probability to this terminal node
    Prob(Node) = Multipler;
    return;
  }
  
  size_t SplitVar = OneTree.SplitVar(Node);
  
  // for splitting j, randomly distribute into left and right
  // using the child node sizes
  if (SplitVar == j)
  {
    double LeftWeight = OneTree.NodeWeight(OneTree.LeftNode(Node));
    double RightWeight = OneTree.NodeWeight(OneTree.RightNode(Node));
    
    Assign_Terminal_Node_Prob_RandomJ(OneTree.LeftNode(Node),
                                      OneTree,
                                      X,
                                      Ncat,
                                      id,
                                      Multipler * LeftWeight / (LeftWeight + RightWeight),
                                      Prob,
                                      j);
      
    
    Assign_Terminal_Node_Prob_RandomJ(OneTree.RightNode(Node),
                                      OneTree,
                                      X,
                                      Ncat,
                                      id,
                                      Multipler * RightWeight / (LeftWeight + RightWeight),
                                      Prob,
                                      j);
  }else{
    // splitting on other variables
    // determine where to go

    double SplitValue = OneTree.SplitValue(Node);
    double xtemp = X( id, SplitVar );      
    bool right = false; 
    
    if ( Ncat(SplitVar) > 1 ) // categorical var 
    {
      
      uvec goright(Ncat(SplitVar) + 1);
      unpack(SplitValue, Ncat(SplitVar) + 1, goright);
      
      if ( goright( (size_t) xtemp ) == 1 )
        right = true;

    }else{ // continuous var 
      
      if (xtemp > SplitValue)
        right = true;
      
    }
    
    // go further down 
    
    if (right)
    {
      Assign_Terminal_Node_Prob_RandomJ(OneTree.RightNode(Node),
                                        OneTree,
                                        X,
                                        Ncat,
                                        id,
                                        Multipler,
                                        Prob,
                                        j);
    }else{
      Assign_Terminal_Node_Prob_RandomJ(OneTree.LeftNode(Node),
                                        OneTree,
                                        X,
                                        Ncat,
                                        id,
                                        Multipler,
                                        Prob,
                                        j);
    }
  }
  
  return;
}

