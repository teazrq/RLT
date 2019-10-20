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

void Reg_Uni_Split_Cat(Reg_Uni_Split_Class& TempSplit,
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
                       bool useobsweight,
                       size_t ncat)
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
    }
  }
  
  sort(cat_reduced.begin(), cat_reduced.end(), reg_cat_reduced_compare);
  
  size_t true_cat = 0;
  for (size_t j = 0; j < cat_reduced.size(); j++)
    if (cat_reduced[j].count) true_cat++;
  
  if (true_cat <= 1)
    return;
  
  // for (size_t j = 0; j < cat_reduced.size(); j++)
      // cat_reduced[j].print();
  
  DEBUG_Rcout << "        --- true_cat " << true_cat << std::endl;
  
  double temp_score = 0;
  
  // rank split, figure out low and high index
  size_t lowindex;
  size_t highindex;
  
  if ( split_gen == 2 or split_gen == 3 )
  {
    reg_move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);
  }else{
    lowindex = 0;
    highindex = true_cat - 2;
  }
  
  size_t best_cat;
  double best_score = -1;
  
  DEBUG_Rcout << "        --- start split with lowindex " << lowindex << " highindex " << highindex << std::endl;
  
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
    
    uvec goright(ncat + 1, fill::zeros); // the first element (category) of goright will always be set to 0 --- go left, but this category does not exist.
 
    for (size_t i = 0; i <= best_cat; i++)
      goright[cat_reduced[i].cat] = 0;
    
    for (size_t i = best_cat + 1; i < true_cat; i++)
      goright[cat_reduced[i].cat] = 1;
      
    for (size_t i = true_cat + 1; i < ncat + 1; i++)
      goright[cat_reduced[i].cat] = 0; //intRand(0, 1); // for empty category, assign randomly
  
    DEBUG_Rcout << "        --- goright is " << goright << std::endl;  
    
    TempSplit.value = pack(ncat + 1, goright);
    
    DEBUG_Rcout << "        --- value is " << TempSplit.value << std::endl;  
    
    TempSplit.score = best_score;
    
    

  }
}

bool reg_cat_reduced_compare(Reg_Cat_Class& a, Reg_Cat_Class& b)
{
  if (a.count == 0 and b.count == 0)
    return 0;
  
  if (a.count > 0 and b.count == 0)
    return 1;
  
  if (a.count == 0 and b.count > 0)
    return 0;
  
  return (a.y < b.y);
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



void reg_move_cat_index(size_t& lowindex, size_t& highindex, std::vector<Reg_Cat_Class>& cat_reduced, size_t true_cat, size_t nmin)
{
  // in this case, we will not be able to control for nmin
  // but extreamly small nmin should not have a high splitting score
  lowindex = 0;
  highindex = true_cat - 2;
  
  if (true_cat == 2) //nothing we can do
    return; 
  
  DEBUG_Rcout << "        --- start moving index with lowindex " << lowindex << " highindex " << highindex << std::endl;
  
  lowindex = 0;
  highindex = true_cat-2;
  size_t lowcount = cat_reduced[0].count;
  size_t highcount = cat_reduced[true_cat-1].count;
  
  // now both low and high index are not tied with the end
  if ( lowcount >= nmin and highcount >= nmin ) // everything is good
    return;
  
  if ( lowcount < nmin and highcount >= nmin ) // only need to fix lowindex
  {
    while( lowcount < nmin and lowindex <= highindex ){
      lowindex++;
      lowcount += cat_reduced[lowindex].count;
    }
    
    if ( lowindex > highindex ) lowindex = highindex;
    
    return;
    DEBUG_Rcout << "        --- case 1 with lowindex " << lowindex << " highindex " << highindex << std::endl;
  }
  
  if ( lowcount >= nmin and highcount < nmin ) // only need to fix highindex
  {
    while( highcount < nmin and lowindex <= highindex ){
      DEBUG_Rcout << "        --- adding " << cat_reduced[highindex].count << " count to highcount " << highcount << std::endl;
      highcount += cat_reduced[highindex].count;
      highindex--;
    }
    
    if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex; // sometimes highindex will be negative and turned into very large number 
    
    DEBUG_Rcout << "        --- case 2 with lowindex " << lowindex << " highindex " << highindex << std::endl;    
    
    return;
  }
  
  if ( lowcount < nmin and highcount < nmin ) // if both need to be fixed, start with one randomly
  {
    if ( intRand(0, 1) )
    { // fix lowindex first
      while( lowcount < nmin and lowindex <= highindex ){
        lowindex++;
        lowcount += cat_reduced[lowindex].count;
      }
      
      if (lowindex > highindex ) lowindex = highindex;
      
      while( highcount < nmin and lowindex <= highindex ){
        highcount += cat_reduced[highindex].count;
        highindex--;
      }
      
      if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex;
      
      DEBUG_Rcout << "        --- case 3 with lowindex " << lowindex << " highindex " << highindex << std::endl;
      return;
      
    }else{ // fix highindex first
      while( highcount < nmin and lowindex <= highindex ){
        highcount += cat_reduced[highindex].count;
        highindex--;
      }
      
      if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex;
      
      while( lowcount < nmin and lowindex <= highindex ){
        lowindex++;
        lowcount += cat_reduced[lowindex].count;
      }
      
      if (lowindex > highindex) lowindex = highindex;
      
      DEBUG_Rcout << "        --- case 4 with lowindex " << lowindex << " highindex " << highindex << std::endl;
      return;
    }
  }
}



