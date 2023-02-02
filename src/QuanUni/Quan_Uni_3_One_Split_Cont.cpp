//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Quantile
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split on a particular variable
void Quan_Uni_Split_Cont(Split_Class& TempSplit,
                        const uvec& obs_id,
                        const vec& x,
                        const vec& Y,
                        const vec& obs_weight,
                        double penalty,
                        size_t split_gen,
                        size_t split_rule,
                        size_t nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl)
{
  size_t N = obs_id.n_elem;

  //arma::vec temp_cut_arma;
  //double temp_cut;
  //size_t temp_ind;
  double temp_score;

  if (split_gen == 1) // random split
  {
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a random cut off
      size_t temp_id = obs_id( rngl.rand_sizet(0,N-1) );
      double temp_cut = x(temp_id);
        
      //temp_cut_arma = x(obs_id( rngl.rand_sizet(0,N-1) )); 
      //temp_cut = temp_cut_arma(0);

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
  
  uvec indices = obs_id(sort_index(x(obs_id))); // this is the sorted obs_id  
  
  // check identical 
  if ( x(indices(0)) == x(indices(N-1)) ) return;  
  
  // set low and high index
  size_t lowindex = 0; // less equal goes to left
  size_t highindex = N - 2;
  
  // alpha is only effective when x can be sorted
  // I need to do some changes to this
  // this will force nmin for each child node
  if (alpha > 0)
  {
    // if (N*alpha > nmin) nmin = (size_t) N*alpha;
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
      RLTcout << "lowindex > highindex... this shouldn't happen." << std::endl;
      return;
    }
  }
  
  /*
    // if there are ties, do further check
    if ( (x(indices(lowindex)) == x(indices(lowindex + 1))) | (x(indices(highindex)) == x(indices(highindex + 1))) )
      move_cont_index(lowindex, highindex, x, indices, nmin);
    
  }else{
    // move index if ties
    while( x(indices(lowindex)) == x(indices(lowindex + 1)) ) lowindex++;
    while( x(indices(highindex)) == x(indices(highindex + 1)) ) highindex--;
    
    //If there is nowhere to split
    if (lowindex > highindex) return;
  }
  */
  
  if (split_gen == 2) // rank split
  {
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a cut off
      size_t temp_ind = rngl.rand_sizet( lowindex, highindex );
      
      // there could be ties here. need to fix this issue. 
      if ( x(indices(temp_ind)) == x(indices(temp_ind+1)) )
      {
        if (rngl.rand_01() > 0.5)
        { // move up
          while( x(indices(temp_ind)) == x(indices(temp_ind+1)) ) temp_ind++;
        }else{ // move down
          while( x(indices(temp_ind)) == x(indices(temp_ind+1)) ) temp_ind--;
        }
      }
      
      if (useobsweight)
        temp_score = reg_uni_cont_score_rank_sub_w(indices, Y, temp_ind, obs_weight);
      else
        temp_score = reg_uni_cont_score_rank_sub(indices, Y, temp_ind);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = (x(indices(temp_ind)) + x(indices(temp_ind+1)))/2 ;
        TempSplit.score = temp_score;
      }
    }
    
    return;
  }
  
  if (split_gen == 3) // best split  
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

