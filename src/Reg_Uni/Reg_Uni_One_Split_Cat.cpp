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

void Reg_Uni_Split_Cat(Uni_Split_Class& TempSplit,
                       uvec& obs_id,
                       const vec& x,
                       const size_t ncat,
                       const vec& Y,
                       const vec& obs_weight,
                       double penalty,
                       int split_gen,
                       int split_rule,
                       int nsplit,
                       size_t nmin,
                       double alpha,
                       bool useobsweight)
{
  DEBUG_Rcout << "        --- Reg_One_Split_Cat with ncat = " << ncat << std::endl;
  
  std::vector<Reg_Cat_Class> cat_reduced(ncat + 1);
  
  for (size_t j = 0; j < cat_reduced.size(); j++)
  {
    cat_reduced[j].cat = j;
  }
  
  if (useobsweight){
    for (size_t i = 0; i < obs_id.size(); i++)
    {
      size_t temp_cat = (size_t) x(obs_id(i));
      cat_reduced[temp_cat].weight += obs_weight(obs_id(i));
      cat_reduced[temp_cat].y += Y(obs_id(i))*obs_weight(obs_id(i));
      cat_reduced[temp_cat].count ++;
    }
  }else{
    for (size_t i = 0; i < obs_id.size(); i++)
    {
      size_t temp_cat = (size_t) x(obs_id(i));
      cat_reduced[temp_cat].y += Y(obs_id(i));
      cat_reduced[temp_cat].count ++;
	    cat_reduced[temp_cat].weight ++;
    }
  }
  
  size_t true_cat = 0;
  for (size_t j = 0; j < cat_reduced.size(); j++)
    if (cat_reduced[j].count) true_cat++;
  
  if (true_cat <= 1)
    return;  
  
  for (size_t j = 0; j < cat_reduced.size(); j++)
      cat_reduced[j].calculate_score();
  
  // this will move the 0 categories to the tail
  sort(cat_reduced.begin(), cat_reduced.end(), cat_reduced_compare);
  
  /*
  sort(cat_reduced.begin(), cat_reduced.begin()+true_cat, cat_reduced_compare_score);
  
  for (size_t j = 0; j < cat_reduced.size(); j++)
      cat_reduced[j].print();
  */
  
  size_t best_cat;
  double temp_score = 0;  
  double best_score = -1;  

  
  // rank split, figure out low and high index
  size_t lowindex = 0;
  size_t highindex = true_cat - 2;  
  
  // alpha is only effective when x can be sorted
  // this will force nmin for each child node
  if (alpha > 0)
  {
    size_t N = obs_id.n_elem;
    
    if (N*alpha > nmin) nmin = (size_t) N*alpha;
    
    move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);
  }

  // start split 
  
  if ( split_gen == 1 or split_gen == 2 )
  {
    for ( int k = 0; k < nsplit; k++ )
    {
      size_t temp_cat = (size_t) intRand(lowindex, highindex);
      
      if (useobsweight)
        temp_score = reg_cat_score_w(cat_reduced, temp_cat, true_cat);
      else
        temp_score = reg_cat_score(cat_reduced, temp_cat, true_cat);
      
      DEBUG_Rcout << "        --- temp_score " << temp_score << std::endl;
      
      if (temp_score > best_score)
      {      
        best_cat = temp_cat;
        best_score = temp_score;
      }
    }
  }else{
    // best split
    
    if (useobsweight)
      reg_cat_score_best_w(cat_reduced, lowindex, highindex, true_cat, best_cat, best_score);
    else
      reg_cat_score_best(cat_reduced, lowindex, highindex, true_cat, best_cat, best_score);
  }
  
  if (best_score > TempSplit.score)
  {
    
    DEBUG_Rcout << "        --- record best split with score " << best_score << " best cut at " << best_cat << " true_cat is " << true_cat << std::endl;
    
    TempSplit.value = record_cat_split(cat_reduced, best_cat, true_cat, ncat);
    
    DEBUG_Rcout << "        --- value is " << TempSplit.value << std::endl;  
    
    TempSplit.score = best_score;
    
  }
}

double reg_cat_score(std::vector<Reg_Cat_Class>& cat_reduced, size_t temp_cat, size_t true_cat)
{
  size_t leftn = 0;
  size_t rightn = 0;  
  double LeftSum = 0;
  double RightSum = 0;
    
  for (size_t i = 0; i <= temp_cat; i++)
  {
    LeftSum += cat_reduced[i].y;
    leftn += cat_reduced[i].count;
  }  
  
  for (size_t i = temp_cat+1; i < true_cat; i++)
  {
    RightSum += cat_reduced[i].y;
    rightn += cat_reduced[i].count;
  }
  
  if (leftn > 0 and rightn > 0)
    return LeftSum*LeftSum/leftn + RightSum*RightSum/rightn;

  return -1;
}


double reg_cat_score_w(std::vector<Reg_Cat_Class>& cat_reduced, size_t temp_cat, size_t true_cat)
{
  double leftw = 0;
  double rightw = 0;
  double LeftSum = 0;
  double RightSum = 0;
  
  for (size_t i = 0; i <= temp_cat; i++)
  {
    LeftSum += cat_reduced[i].y;
    leftw += cat_reduced[i].weight;
  }  
  
  for (size_t i = temp_cat+1; i < true_cat; i++)
  {
    RightSum += cat_reduced[i].y;
    rightw += cat_reduced[i].weight;
  }
  
  if (leftw > 0 and rightw > 0)
    return LeftSum*LeftSum/leftw + RightSum*RightSum/rightw;
    
  return -1;
}

void reg_cat_score_best(std::vector<Reg_Cat_Class>& cat_reduced, size_t lowindex, size_t highindex, size_t true_cat, size_t& best_cat, double& best_score)
{

  size_t leftn = 0;
  size_t rightn = 0;  
  double LeftSum = 0;
  double RightSum = 0;
    
  for (size_t i = 0; i <= lowindex; i++)
  {
    LeftSum += cat_reduced[i].y;
    leftn += cat_reduced[i].count;
  }  
  
  for (size_t i = lowindex+1; i < true_cat; i++)
  {
    RightSum += cat_reduced[i].y;
    rightn += cat_reduced[i].count;
  }
  
  best_cat = lowindex;
  best_score = LeftSum*LeftSum/leftn + RightSum*RightSum/rightn;
  
  for (size_t i = lowindex + 1; i <= highindex; i++)
  {
    LeftSum += cat_reduced[i].y;
    leftn += cat_reduced[i].count;
    
    RightSum -= cat_reduced[i].y;
    rightn -= cat_reduced[i].count;
    
    double temp_score = LeftSum*LeftSum/leftn + RightSum*RightSum/rightn;
    
    if (temp_score > best_score)
    {
      best_cat = i;
      best_score = temp_score;
    }
  }
}


void reg_cat_score_best_w(std::vector<Reg_Cat_Class>& cat_reduced, size_t lowindex, size_t highindex, size_t true_cat, size_t& best_cat, double& best_score)
{
  
  double leftw = 0;
  double rightw = 0;  
  double LeftSum = 0;
  double RightSum = 0;
  
  for (size_t i = 0; i <= lowindex; i++)
  {
    LeftSum += cat_reduced[i].y;
    leftw += cat_reduced[i].weight;
  }  
  
  for (size_t i = lowindex+1; i < true_cat; i++)
  {
    RightSum += cat_reduced[i].y;
    rightw += cat_reduced[i].weight;
  }
  
  best_cat = lowindex;
  best_score = LeftSum*LeftSum/leftw + RightSum*RightSum/rightw;
  
  for (size_t i = lowindex + 1; i <= highindex; i++)
  {
    LeftSum += cat_reduced[i].y;
    leftw += cat_reduced[i].count;
    
    RightSum -= cat_reduced[i].y;
    rightw -= cat_reduced[i].count;
    
    double temp_score = LeftSum*LeftSum/leftw + RightSum*RightSum/rightw;
    
    if (temp_score > best_score)
    {
      best_cat = i;
      best_score = temp_score;
    }
  }
}




