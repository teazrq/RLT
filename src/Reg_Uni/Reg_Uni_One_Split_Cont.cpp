//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../regForest.h"

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Split_Cont(Uni_Split_Class& TempSplit, 
                        uvec& obs_id,
                        const vec& x,
                        const vec& Y,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        size_t nmin, 
                        double alpha,
                        vec& obs_weight,
                        bool useobsweight)
{
  size_t N = obs_id.n_elem;

  arma::vec temp_cut_arma;
  double temp_cut;
  size_t temp_ind;
  double temp_score;

  if (split_gen == 1) // random split
  {
    DEBUG_Rcout << "      --- Reg_One_Split_Cont with " << nsplit << " random split " << std::endl;
    
    for (int k = 0; k < nsplit; k++)
    {
      // generate a random cut off
      temp_cut_arma = x(obs_id( (size_t) intRand(0, N-1) ));
      temp_cut = temp_cut_arma(0);
      
      if (useobsweight)
        temp_score = reg_cont_score_at_cut_w(obs_id, x, Y, temp_cut, obs_weight);
      else
        temp_score = reg_cont_score_at_cut(obs_id, x, Y, temp_cut);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = temp_cut;
        TempSplit.score = temp_score;
      }
    }
    
    DEBUG_Rcout << "      --- Best cut off at " << TempSplit.value << " with score " << TempSplit.score << std::endl;
    return;
  }
    
  // alpha is only effective when x can be sorted
  if (N*alpha > nmin) nmin = (size_t) N*alpha;
  
  uvec indices = obs_id(sort_index(x(obs_id))); // this is the sorted obs_id

  // check identical 
  if ( x(indices(0)) == x(indices(N-1)) ) return;
  
  // set low and high index
  size_t lowindex = nmin - 1; // less equal goes to left
  size_t highindex = N - nmin - 1;
  
  // if there are ties, do further check
  if ( (x(indices(lowindex)) == x(indices(lowindex + 1))) | (x(indices(highindex)) == x(indices(highindex + 1))) )
    move_cont_index(lowindex, highindex, x, indices, nmin);
  
  DEBUG_Rcout << "      --- lowindex " << lowindex << " highindex " << highindex << std::endl;
  
  if (split_gen == 2) // rank split
  {
    DEBUG_Rcout << "      --- Reg_One_Split_Cont with " << nsplit << " rank split " << std::endl;
    
    for (int k = 0; k < nsplit; k++)
    {
      // generate a cut off
      temp_ind = intRand(lowindex, highindex);
      
      if (useobsweight)
        temp_score = reg_cont_score_at_index_w(indices, Y, temp_ind, obs_weight);
      else
        temp_score = reg_cont_score_at_index(indices, Y, temp_ind);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = (x(indices(temp_ind)) + x(indices(temp_ind+1)))/2 ;
        TempSplit.score = temp_score;
      }
    }
    
    DEBUG_Rcout << "      --- Rank cut off at " << TempSplit.value << " with score " << TempSplit.score << std::endl;
    return;
  }
  
  
  if (split_gen == 3) // best split  
  {
    DEBUG_Rcout << "      --- Reg_One_Split_Cont with best split, total sample " << x.size() << std::endl;
    
    if (useobsweight)
      reg_cont_score_best_w(indices, x, Y, lowindex, highindex, TempSplit.value, TempSplit.score, obs_weight);
    else
      reg_cont_score_best(indices, x, Y, lowindex, highindex, TempSplit.value, TempSplit.score);

      
    DEBUG_Rcout << "      --- Best cut off at " << TempSplit.value << " with score " << TempSplit.score << std::endl;
    
    return;
    
  }
}


double reg_cont_score_at_cut(uvec& obs_id,
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
    if ( x(obs_id(i)) <= a_random_cut )
    {
      LeftCount++;
      LeftSum += Y(obs_id(i));
    }else{
      RightSum += Y(obs_id(i));
    }
  }
  
  if (LeftCount > 0 && LeftCount < N)
    return LeftSum*LeftSum/LeftCount + RightSum*RightSum/(N - LeftCount);
  
  return -1;
}


double reg_cont_score_at_cut_w(uvec& obs_id,
                          const vec& x,
                          const vec& Y,
                          double a_random_cut,
                          vec& obs_weight)
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




double reg_cont_score_at_index(uvec& indices,
                          const vec& Y,
                          size_t a_random_ind)
{
  size_t N = indices.size();
  
  double LeftSum = 0;
  double RightSum = 0;
  
  for (size_t i = 0; i <= a_random_ind; i++)
    LeftSum += Y(indices(i));
  
  for (size_t i = a_random_ind+1; i < N; i++)  
    RightSum += Y(indices(i));

  return LeftSum*LeftSum/a_random_ind + RightSum*RightSum/(N - a_random_ind);
}



double reg_cont_score_at_index_w(uvec& indices,
                            const vec& Y,
                            size_t a_random_ind,
                            vec& obs_weight)
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



void reg_cont_score_best(uvec& indices,
                    const vec& x,
                    const vec& Y,
                    size_t lowindex, 
                    size_t highindex, 
                    double& temp_cut, 
                    double& temp_score)
{
  DEBUG_Rcout << "      --- Best score with no weights --- " << std::endl;
  
  double score = 0;
  
  size_t N = indices.size();
  
  double LeftSum = 0;
  double RightSum = 0;
  // size_t LeftCount = lowindex + 1;
  
  for (size_t i = 0; i <= lowindex; i++)
    LeftSum += Y(indices(i));
  
  for (size_t i = lowindex+1; i < N; i++)
    RightSum += Y(indices(i));

  for (size_t i = lowindex; i <= highindex; i++)
  {
    
    while (x(indices(i)) == x(indices(i+1))){
      i++;
      
      LeftSum += Y(indices(i));
      RightSum -= Y(indices(i));
    }
    
    score = LeftSum*LeftSum/(i + 1) + RightSum*RightSum/(N - i - 1);
    
    if (score > temp_score)
    {
      temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
      temp_score = score;
    }
    
    if (i + 1 <= highindex)
    {
      LeftSum += Y(indices(i+1));
      RightSum -= Y(indices(i+1));
    }
  }
}

void reg_cont_score_best_w(uvec& indices,
                      const vec& x,
                      const vec& Y,
                      size_t lowindex, 
                      size_t highindex, 
                      double& temp_cut, 
                      double& temp_score,
                      vec& obs_weight)
{
  DEBUG_Rcout << "      --- Best score with weights --- " << std::endl;
  double score = 0;
  
  size_t N = indices.size();
  
  double LeftSum = 0;
  double RightSum = 0;
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i <= lowindex; i++)
  {
    LeftSum += Y(indices(i))*obs_weight(indices(i));
    Left_w += obs_weight(indices(i));
  }

  for (size_t i = lowindex+1; i < N; i++)
  {
    RightSum += Y(indices(i))*obs_weight(indices(i));
    Right_w += obs_weight(indices(i));
  }
  
  for (size_t i = lowindex; i <= highindex; i++)
  {
    while (x(indices(i)) == x(indices(i+1))){
      i++;
      
      LeftSum += Y(indices(i))*obs_weight(indices(i));
      RightSum -= Y(indices(i))*obs_weight(indices(i));
      
      Left_w += obs_weight(indices(i));
      Right_w += obs_weight(indices(i));
    }
    
    score = LeftSum*LeftSum/Left_w + RightSum*RightSum/Right_w;
    
    if (score > temp_score)
    {
      temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
      temp_score = score;
    }
    
    if (i + 1 <= highindex)
    {
      LeftSum += Y(indices(i+1))*obs_weight(indices(i+1));
      RightSum -= Y(indices(i+1))*obs_weight(indices(i+1));
      
      Left_w += obs_weight(indices(i+1));
      Right_w += obs_weight(indices(i+1));
    }
  }
}