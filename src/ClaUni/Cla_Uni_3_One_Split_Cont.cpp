//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split on a particular variable
void Cla_Uni_Split_Cont(Split_Class& TempSplit,
                        const uvec& obs_id,
                        const vec& x,
                        const uvec& Y,
                        const vec& obs_weight,
                        const size_t nclass,
                        double penalty,
                        size_t split_gen,
                        size_t split_rule,
                        size_t nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl)
{

  size_t N = obs_id.n_elem;
  
  double temp_score;
  
  if (split_gen == 1) // random split
  {
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a random cut off
      size_t temp_id = obs_id( rngl.rand_sizet(0,N-1) );
      double temp_cut = x(temp_id);
      
      // calculate score
      if (useobsweight)
        temp_score = cla_uni_cont_score_cut_sub_w(obs_id, x, Y, nclass, temp_cut, obs_weight);
      else
        temp_score = cla_uni_cont_score_cut_sub(obs_id, x, Y, nclass, temp_cut);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = temp_cut;
        TempSplit.score = temp_score;
      }
    }
    
    return;
  }
  
  // indices is obs_id sorted based on x values
  uvec indices = obs_id(sort_index(x(obs_id)));
  
  // check identical
  if ( x(indices(0)) == x(indices(N-1)) ) return;
  
  // set low and high index
  size_t lowindex = 0; // less equal goes to left
  size_t highindex = N - 2;
  
  // alpha is only effective when x can be sorted 
  // this will force a portion of alpha for each child node 
  if (alpha > 0)
  {
    // size on each side
    size_t nmin = N*alpha < 1 ? 1 : N*alpha;
    
    lowindex = nmin-1; // less equal goes to left
    highindex = N - nmin - 1;
  }
  
  // if ties, move index to better locations
  if ( x(indices(lowindex)) == x(indices(lowindex+1)) or 
         x(indices(highindex)) == x(indices(highindex+1)) )
  {
    check_cont_index_sub(lowindex, highindex, x, indices);
    
    if (lowindex > highindex)
    {
      RLTcout << "lowindex > highindex... this shouldn't happen." << std::endl;
      return;
    }
  }

  // rank split
  if (split_gen == 2)
  {
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a cut off
      size_t temp_ind = rngl.rand_sizet( lowindex, highindex );
      
      // there could be ties here. move up or down
      if ( x(indices(temp_ind)) == x(indices(temp_ind+1)) )
      {
        if (rngl.rand_01() > 0.5)
        { // move up
          while( x(indices(temp_ind)) == x(indices(temp_ind+1)) ) temp_ind++;
        }else{ // move down
          while( x(indices(temp_ind)) == x(indices(temp_ind+1)) ) temp_ind--;
        }
      }
      
      // calculate scores
      if (useobsweight)
        temp_score = cla_uni_cont_score_rank_sub_w(indices, Y, nclass, temp_ind, obs_weight);
      else
        temp_score = cla_uni_cont_score_rank_sub(indices, Y, nclass, temp_ind);
      
      // record
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
      cla_uni_cont_score_best_sub_w(indices, x, Y, nclass, lowindex, highindex, 
                                    TempSplit.value, TempSplit.score, obs_weight);
    else
      cla_uni_cont_score_best_sub(indices, x, Y, nclass, lowindex, highindex, 
                                  TempSplit.value, TempSplit.score);
    return;
  }
}


//For rank split
double cla_uni_cont_score_rank_sub(uvec& indices,
                                   const uvec& Y,
                                   size_t nclass,
                                   size_t a_random_ind)
{
  size_t N = indices.size();
  
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);
  
  //Count the number of observations with a smaller or equal index
  for (size_t i = 0; i <= a_random_ind; i++)
    LeftSum(Y(indices(i)))++;
  
  //Count other observations
  for (size_t i = a_random_ind+1; i < N; i++)
    RightSum(Y(indices(i)))++;
  
  return accu( square(LeftSum) ) / (a_random_ind + 1) + 
         accu( square(RightSum) ) / (N - a_random_ind - 1);
}

//For weighted rank split
double cla_uni_cont_score_rank_sub_w(uvec& indices,
                                     const uvec& Y,
                                     size_t nclass,
                                     size_t a_random_ind,
                                     const vec& obs_weight)
{
  size_t N = indices.size();
  size_t subj;
  
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i <= a_random_ind; i++){
    subj = indices(i);
    LeftSum(Y(subj)) += obs_weight(subj);
    Left_w += obs_weight(subj);
  }
  
  for (size_t i = a_random_ind+1; i < N; i++){
    subj = indices(i);
    RightSum(Y(subj)) += obs_weight(subj);
    Right_w += obs_weight(subj);
  }
  
  return accu( square(LeftSum) ) / Left_w + 
         accu( square(RightSum) ) / Right_w;
}


double cla_uni_cont_score_cut_sub(const uvec& obs_id,
                                  const vec& x,
                                  const uvec& Y,
                                  size_t nclass,
                                  double a_random_cut)
{
  size_t N = obs_id.size();
  size_t subj;
  
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);
  size_t LeftCount = 0;
  
  for (size_t i = 0; i < N; i++)
  {
    subj = obs_id(i);

    //If x is less than the random cut, go left
    if ( x(subj) <= a_random_cut )
    {
      LeftCount++;
      LeftSum(Y(subj))++;
    }else{
      RightSum(Y(subj))++;
    }
  }
  
  // RLTcout << "\n left is \n" << LeftSum << "\n right is \n" << RightSum << "\n" <<std::endl; 
  
  // if there are some observations in each node
  if (LeftCount > 0 && LeftCount < N)
  {
    return accu( square(LeftSum) ) / LeftCount + 
           accu( square(RightSum) ) / (N - LeftCount);
  }

  return -1;
}

double cla_uni_cont_score_cut_sub_w(const uvec& obs_id,
                                    const vec& x,
                                    const uvec& Y,
                                    size_t nclass,
                                    double a_random_cut,
                                    const vec& obs_weight)
{
  size_t N = obs_id.size();
  
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i < N; i++)
  {
    size_t subj = obs_id(i);
    double wi = obs_weight(subj);

    if ( x(subj) <= a_random_cut )
    {
      Left_w += wi;
      LeftSum(Y(subj)) += wi;
    }else{
      Right_w += wi;
      RightSum(Y(subj)) += wi;
    }
  }
  
  if (Left_w > 0 && Right_w > 0)
  {
    return sum( square(LeftSum) ) / Left_w + 
           sum( square(RightSum) ) / Right_w;
  }
  
  return -1;
}


//For best split
void cla_uni_cont_score_best_sub(uvec& indices,
                                 const vec& x,
                                 const uvec& Y,
                                 size_t nclass,
                                 size_t lowindex, 
                                 size_t highindex, 
                                 double& temp_cut, 
                                 double& temp_score)
{
  double score = 0;
  size_t N = indices.size();
  
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);

  //Find left or right of the lowindex to start
  for (size_t i = 0; i <= lowindex; i++)
    LeftSum(Y(indices(i)))++;
  
  for (size_t i = lowindex+1; i < N; i++)
    RightSum(Y(indices(i)))++;

  //Trying the other splits
  for (size_t i = lowindex; i <= highindex; i++)
  {
    //If there is a tie
    while (x(indices(i)) == x(indices(i+1))){
      i++;
      
      //Adjust sums
      LeftSum(Y(indices(i)))++;
      RightSum(Y(indices(i)))--;
    }
    
    //Calculate score
    score = accu( square(LeftSum) ) / (i + 1) + 
            accu( square(RightSum) ) / (N - i - 1);
    
    //If the score has improved, find cut and set new score
    if (score > temp_score)
    {
      temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
      temp_score = score;
    }
    
    //Adjust sums
    if (i + 1 <= highindex)
    {
      LeftSum(Y(indices(i+1)))++;
      RightSum(Y(indices(i+1)))--;
    }
  }
}


//For best split weighted
void cla_uni_cont_score_best_sub_w(uvec& indices,
                                   const vec& x,
                                   const uvec& Y,
                                   size_t nclass,
                                   size_t lowindex, 
                                   size_t highindex, 
                                   double& temp_cut, 
                                   double& temp_score,
                                   const vec& obs_weight)
{
  double score = 0;
  
  size_t N = indices.size();
  size_t subj;
  
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i <= lowindex; i++)
  {
    subj = indices(i);
    LeftSum(Y(subj)) += obs_weight(subj);
    Left_w += obs_weight(subj);
  }
  
  for (size_t i = lowindex+1; i < N; i++)
  {
    subj = indices(i);
    RightSum(Y(subj)) += obs_weight(subj);
    Right_w += obs_weight(subj);
  }
  
  for (size_t i = lowindex; i <= highindex; i++)
  {
    while (x(indices(i)) == x(indices(i+1))){
      i++;
      subj = indices(i);
      LeftSum(Y(subj)) += obs_weight(subj);
      RightSum(Y(subj)) -= obs_weight(subj);
      
      Left_w += obs_weight(subj);
      Right_w -= obs_weight(subj);
    }
    
    score = accu( square(LeftSum) ) / Left_w + 
            accu( square(RightSum) ) / Right_w;
    
    if (score > temp_score)
    {
      temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
      temp_score = score;
    }
    
    if (i + 1 <= highindex)
    {
      subj = indices(i+1);
      
      LeftSum(Y(subj)) += obs_weight(subj);
      RightSum(Y(subj)) -= obs_weight(subj);
      
      Left_w += obs_weight(subj);
      Right_w -= obs_weight(subj);
    }
  }
}





