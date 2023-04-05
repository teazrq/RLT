//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split on a particular variable
void Reg_Uni_Split_Cont(Split_Class& TempSplit,
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
        temp_score = reg_uni_cont_score_cut_sub_w(obs_id, x, Y, temp_cut, obs_weight);
      else
        temp_score = reg_uni_cont_score_cut_sub(obs_id, x, Y, temp_cut);
      
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
    size_t nmin = (size_t) N*alpha;
    if (nmin < 1) nmin = 1;
    
    lowindex = nmin-1; // less equal goes to left
    highindex = N - nmin - 1;
  }
  
  // if ties, move index to better locations
  if ( x(indices(lowindex)) == x(indices(lowindex+1)) or x(indices(highindex)) == x(indices(highindex+1)) )
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

//Calculate a score at a random cut
double reg_uni_cont_score_cut_sub(const uvec& obs_id,
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


double reg_uni_cont_score_cut_sub_w(const uvec& obs_id,
                                    const vec& x,
                                    const vec& Y,
                                    double a_random_cut,
                                    const vec& obs_weight)
{
  size_t N = obs_id.size();
  
  double LeftSum = 0;
  double RightSum = 0;
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i < N; i++)
  {
    double wi = obs_weight(obs_id(i));
    
    if ( x(obs_id(i)) <= a_random_cut )
    {
      Left_w += wi;
      LeftSum += Y(obs_id(i))*wi;
    }else{
      Right_w += wi;
      RightSum += Y(obs_id(i))*wi;
    }
  }
  
  if (Left_w > 0 && Right_w < N)
    return LeftSum*LeftSum/Left_w + RightSum*RightSum/Right_w;
  
  return -1;
}



//For rank split
double reg_uni_cont_score_rank_sub(uvec& indices,
                                   const vec& Y,
                                   size_t a_random_ind)
{
  size_t N = indices.size();
  
  double LeftSum = 0;
  double RightSum = 0;
  
  //Count the number of observations with a smaller or equal index
  for (size_t i = 0; i <= a_random_ind; i++)
    LeftSum += Y(indices(i));
  
  //Count the other observations
  for (size_t i = a_random_ind+1; i < N; i++)  
    RightSum += Y(indices(i));

  return LeftSum*LeftSum/(a_random_ind + 1) + RightSum*RightSum/(N - a_random_ind - 1);
}



double reg_uni_cont_score_rank_sub_w(uvec& indices,
                                     const vec& Y,
                                     size_t a_random_ind,
                                     const vec& obs_weight)
{
  size_t N = indices.size();
  
  double LeftSum = 0;
  double RightSum = 0;
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i <= a_random_ind; i++){
    LeftSum += Y(indices(i))*obs_weight(indices(i));
    Left_w += obs_weight(indices(i));
  }
    
  
  for (size_t i = a_random_ind+1; i < N; i++){
    RightSum += Y(indices(i))*obs_weight(indices(i));
    Right_w += obs_weight(indices(i));
  }
    
  return LeftSum*LeftSum/Left_w + RightSum*RightSum/Right_w;
}


//For best split
void reg_uni_cont_score_best_sub(uvec& indices,
                                 const vec& x,
                                 const vec& Y,
                                 size_t lowindex, 
                                 size_t highindex, 
                                 double& temp_cut, 
                                 double& temp_score)
{

  double score = 0;
  
  size_t N = indices.size();
  
  double LeftSum = 0;
  double RightSum = 0;

  //Find left or right of the lowindex to start
  for (size_t i = 0; i <= lowindex; i++)
    LeftSum += Y(indices(i));
  
  for (size_t i = lowindex+1; i < N; i++)
    RightSum += Y(indices(i));

  //Trying the other splits
  for (size_t i = lowindex; i <= highindex; i++)
  {
    
    //If there is a tie
    while (x(indices(i)) == x(indices(i+1))){
      i++;
      
      //Adjust sums
      LeftSum += Y(indices(i));
      RightSum -= Y(indices(i));
    }
    
    //Calculate score
    score = LeftSum*LeftSum/(i + 1) + RightSum*RightSum/(N - i - 1);
    
    //If the score has improved, find cut and set new score
    if (score > temp_score)
    {
      temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
      temp_score = score;
    }
    
    //Adjust sums
    if (i + 1 <= highindex)
    {
      LeftSum += Y(indices(i+1));
      RightSum -= Y(indices(i+1));
    }
  }
}


void reg_uni_cont_score_best_sub_w(uvec& indices,
                                   const vec& x,
                                   const vec& Y,
                                   size_t lowindex, 
                                   size_t highindex, 
                                   double& temp_cut, 
                                   double& temp_score,
                                   const vec& obs_weight)
{
  double score = 0;
  
  size_t N = indices.size();
  size_t subj;
  
  double LeftSum = 0;
  double RightSum = 0;
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i <= lowindex; i++)
  {
    subj = indices(i);
    LeftSum += Y(subj)*obs_weight(subj);
    Left_w += obs_weight(subj);
  }

  for (size_t i = lowindex+1; i < N; i++)
  {
    subj = indices(i);
    RightSum += Y(subj)*obs_weight(subj);
    Right_w += obs_weight(subj);
  }
  
  for (size_t i = lowindex; i <= highindex; i++)
  {
    while (x(indices(i)) == x(indices(i+1))){
      i++;
      subj = indices(i);
      LeftSum += Y(subj)*obs_weight(subj);
      RightSum -= Y(subj)*obs_weight(subj);
      
      Left_w += obs_weight(subj);
      Right_w -= obs_weight(subj);
    }
    
    score = LeftSum*LeftSum/Left_w + RightSum*RightSum/Right_w;
    
    if (score > temp_score)
    {
      temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
      temp_score = score;
    }
    
    if (i + 1 <= highindex)
    {
      subj = indices(i+1);
      
      LeftSum += Y(subj)*obs_weight(subj);
      RightSum -= Y(subj)*obs_weight(subj);
      
      Left_w += obs_weight(subj);
      Right_w -= obs_weight(subj);
    }
  }
}