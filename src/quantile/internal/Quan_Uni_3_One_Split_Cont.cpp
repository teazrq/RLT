//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Quantile
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split on a particular variable
void Quan_Uni_Split_Cont(Split_Class& TempSplit,
                        const uvec& obs_id,
                        const vec& x,
                        const vec& Y,
                        const vec& obs_weight,
                        double penalty,                        size_t split_rule,
                        size_t nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl)
{
  size_t N = obs_id.n_elem;

  double temp_score;

  if (nsplit > 0) // random split
  {
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a random cut off
      size_t temp_id = obs_id( rngl.rand_sizet(0,N-1) );
      double temp_cut = x(temp_id);

      // calculate score
      if (useobsweight)
        temp_score = reg_uni_cont_score_cut_sub_w(obs_id, x, Y, temp_cut, obs_weight);
      else
        temp_score = quan_uni_cont_score_cut_sub(obs_id, x, Y, temp_cut);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = temp_cut;
        TempSplit.score = temp_score;
      }
    }
    
    return;
  }
  
  uvec indices = obs_id(stable_sort_index(x(obs_id))); // this is the sorted obs_id
  
  // check identical 
  if ( x(indices(0)) == x(indices(N-1)) ) return;  
  
  // set low and high index
  size_t lowindex = 0; // less equal goes to left
  size_t highindex = N - 2;
  
  // alpha enforces a minimum node size for each child node (fraction of N)
  if (alpha > 0)
  {
    size_t nmin = (size_t) N*alpha;
    if (nmin < 1) nmin = 1;
    
    lowindex = nmin-1; // less equal goes to left
    highindex = N - nmin - 1;
  }
  
  // if ties
  // move index to better locations
  if ( x(indices(lowindex)) == x(indices(lowindex+1)) or x(indices(highindex)) == x(indices(highindex+1)) )
  {
    check_cont_index_sub(lowindex, highindex, x, indices);
    
    if (lowindex > highindex)
    {
      return;
    }
  }
  
  if (nsplit == 0) // best split  
  {
    // get score
    if (useobsweight)
      reg_uni_cont_score_best_sub_w(indices, x, Y, lowindex, highindex, TempSplit.value, TempSplit.score, obs_weight);
    else
      reg_uni_cont_score_best_sub(indices, x, Y, lowindex, highindex, TempSplit.value, TempSplit.score);
    
    return;
  }
  
}

//Calculate a KS score at a random cut
double quan_uni_cont_score_cut_sub(const uvec& obs_id,
                                  const vec& x,
                                  const vec& Y,
                                  double a_random_cut)
{
  size_t N = obs_id.size();
  
  double LeftSum = 0;
  double RightSum = 0;
  size_t LeftCount = 0;
  
  for (size_t i = 0; i < N; i++)
  {
    //If x is less than the random cut, go left
    if ( x(obs_id(i)) <= a_random_cut )
    {
      LeftCount++;
      LeftSum += Y(obs_id(i));
    }else{
      RightSum += Y(obs_id(i));
    }
  }
  
  // if there are some observations in each node
  if (LeftCount > 0 && LeftCount < N)
    return LeftSum*LeftSum/LeftCount + RightSum*RightSum/(N - LeftCount);
  
  return -1;
}

