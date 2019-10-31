//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Split_Cat(Uni_Split_Class& TempSplit,
                       uvec& obs_id,
                       const vec& x,
                       const uvec& Y, // Y is collapsed
                       const uvec& Censor, // Censor is collapsed
                       double penalty,
                       int split_gen,
                       int split_rule,
                       int nsplit,
                       size_t nmin,
                       double alpha,
                       vec& obs_weight,
                       bool useobsweight,
                       size_t NFail,
                       int failforce,
                       size_t ncat)
{
    DEBUG_Rcout << "        --- Surv_One_Split_Cat with ncat = " << ncat << std::endl;
    
    if (NFail == 0)
      return; 
    
    std::vector<Surv_Cat_Class> cat_reduced(ncat + 1);
    
    for (size_t j = 0; j < cat_reduced.size(); j++)
        cat_reduced[j].initiate(j, NFail);    
    
    if (useobsweight){
        
        DEBUG_Rcout << "        --- weighted cat split for surv not done yet " << std::endl;
        
    }else{
     
        for (size_t i = 0; i < obs_id.size(); i++)
        {
            size_t temp_cat = (size_t) x(obs_id(i));
            cat_reduced[temp_cat].weight++;
            cat_reduced[temp_cat].count++;
            
            if (Censor(i) == 1)
                cat_reduced[temp_cat].FailCount(Y(i))++; 
            else
                cat_reduced[temp_cat].CensorCount(Y(i))++;
        }
    }
    
    size_t true_cat = 0;
  
    for (size_t j = 0; j < cat_reduced.size(); j++)
        if (cat_reduced[j].count) true_cat++;

    if (true_cat <= 1)  // nothing to split
        return;        
    
    // if only two categories, then split on middle
    
    if (true_cat == 2)
    {
        sort(cat_reduced.begin(), cat_reduced.end(), cat_reduced_compare);
        
        // weighted version and count version are combined into the same function
        TempSplit.score = surv_cat_score(cat_reduced, 0, true_cat, NFail, split_rule, useobsweight);
        TempSplit.value = record_cat_split(cat_reduced, 0, true_cat, ncat);
        
        return;
    }
    
    // if more than 2 categories, 
    
    // calculate the cHaz for each category (this is for sorting later on)
    
    for (size_t j = 1; j < cat_reduced.size(); j++)
    {
      cat_reduced[j].calculate_cHaz(NFail);
    }
    
    sort(cat_reduced.begin(), cat_reduced.end(), cat_reduced_compare);
    
    size_t temp_cat = 0;
    double temp_score = -1;
          
    if ( split_gen == 1 or split_gen == 2 )
    {
        for ( int k = 0; k < nsplit; k++ )
        {
          size_t timepoint = intRand(1, NFail); // randomly select a timepoint for sorting
          
          for (size_t j = 1; j < true_cat; j++)
            cat_reduced[j].set_score(timepoint);
          
          // sort based on this new score 
          sort(cat_reduced.begin(), cat_reduced.begin() + true_cat, cat_reduced_compare);
          
          // get low and high index since the categories are re-ordered
          size_t lowindex = 0;
          size_t highindex = true_cat - 2;
          
          if ( split_gen == 2 )
            move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);

          // generate a random split 
          size_t temp_cat = (size_t) intRand(lowindex, highindex);
          
          // calculate score of this split (weight version and count version are combined)
          temp_score = surv_cat_score(cat_reduced, temp_cat, true_cat, NFail, split_rule, useobsweight);
          
          DEBUG_Rcout << "        --- temp_score " << temp_score << std::endl;
          
          if (temp_score > TempSplit.score)
          {      
            TempSplit.value = record_cat_split(cat_reduced, temp_cat, true_cat, ncat);
            TempSplit.score = temp_score;
          }
        }
    }else{
      // best split
      
      for (size_t j = 1; j < true_cat; j++)
        cat_reduced[j].set_score_ccHaz();
      
      // sort based on this new score 
      sort(cat_reduced.begin(), cat_reduced.begin() + true_cat, cat_reduced_compare);      
      
      // get low and high index since the categories are re-ordered
      size_t lowindex = 0;
      size_t highindex = true_cat - 2;
      move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);
      
      // calculate score at each split (combined weight/count versions)
      surv_cat_score_best(cat_reduced, lowindex, highindex, true_cat, temp_cat, temp_score, NFail, split_rule, useobsweight);
      
      TempSplit.value = record_cat_split(cat_reduced, temp_cat, true_cat, ncat);
      TempSplit.score = temp_score;
    }
}


double surv_cat_score(std::vector<Surv_Cat_Class>& cat_reduced, 
                      size_t temp_cat, 
                      size_t true_cat,
                      size_t NFail, 
                      int split_rule,
                      bool useobsweight)
{
  vec Left_Count_Fail(NFail+1, fill::zeros);
  vec Left_Count_Censor(NFail+1, fill::zeros);
  vec Right_Count_Fail(NFail+1, fill::zeros);
  vec Right_Count_Censor(NFail+1, fill::zeros);
  
  double LeftW = 0;     
  double RightW = 0;
  
  // initiate the failure and censoring counts
  for (size_t i = 0; i<= temp_cat; i++)
  {
    Left_Count_Fail += cat_reduced[i].FailCount;
    Left_Count_Censor += cat_reduced[i].CensorCount;
    LeftW += cat_reduced[i].weight;
  }
  
  for (size_t i = temp_cat+1; i < true_cat; i++)
  {
    Right_Count_Fail += cat_reduced[i].FailCount;
    Right_Count_Censor += cat_reduced[i].CensorCount;
    RightW += cat_reduced[i].weight;
  }
  
  if (split_rule == 1)
    return logrank_w(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, LeftW, LeftW + RightW, NFail, useobsweight);
  
  if (split_rule == 2)
    return suplogrank_w(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, LeftW, LeftW + RightW, NFail, useobsweight);
  
  Rcout << "      --- splitting rule not implemented yet " << std::endl;
}

void surv_cat_score_best(std::vector<Surv_Cat_Class>& cat_reduced,
                         size_t lowindex,
                         size_t highindex,
                         size_t true_cat,
                         size_t& temp_cat,
                         double& temp_score,
                         size_t NFail,
                         int split_rule, 
                         bool useobsweight)
{
  double score = 0;
  
  double LeftW = 0;
  double RightW = 0;  
  
  vec Left_Count_Fail(NFail+1, fill::zeros);
  vec Left_Count_Censor(NFail+1, fill::zeros);
  vec Right_Count_Fail(NFail+1, fill::zeros);
  vec Right_Count_Censor(NFail+1, fill::zeros);
  
  // initiate the failure and censoring counts
  for (size_t i = 0; i<= lowindex; i++)
  {
    Left_Count_Fail += cat_reduced[i].FailCount;
    Left_Count_Censor += cat_reduced[i].CensorCount;
    LeftW += cat_reduced[i].weight;
  }
  
  for (size_t i = lowindex+1; i < true_cat; i++)
  {
    Right_Count_Fail += cat_reduced[i].FailCount;
    Right_Count_Censor += cat_reduced[i].CensorCount;
    RightW += cat_reduced[i].weight;
  }
  
  for (size_t i = lowindex; i <= highindex; i++)
  {
    if (split_rule == 1)
      score = logrank_w(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, LeftW, LeftW + RightW, NFail, useobsweight);
    
    if (split_rule == 2)
      score = suplogrank_w(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, LeftW, LeftW + RightW, NFail, useobsweight);
    
    if (score > temp_score)
    {
      temp_cat = i;
      temp_score = score;
      //Rcout << "      --- update score at cut " << i << " score " << temp_score << std::endl;
      //Rcout << "      --- leftW rightW: " << LeftW << " " << RightW << temp_score << std::endl;
      //Rcout << "      --- data \n" << join_rows(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor) << std::endl;
    }
    
    if (i + 1 <= highindex)
    {
      Left_Count_Fail += cat_reduced[i+1].FailCount;
      Left_Count_Censor += cat_reduced[i+1].CensorCount;
      LeftW += cat_reduced[i+1].weight;
      
      Right_Count_Fail -= cat_reduced[i+1].FailCount;
      Right_Count_Censor -= cat_reduced[i+1].CensorCount;
      RightW -= cat_reduced[i+1].weight;
    }
  }
}


double logrank_w(vec& Left_Count_Fail,
                 vec& Left_Count_Censor,
                 vec& Right_Count_Fail,
                 vec& Right_Count_Censor,
                 double LeftW,
                 double W,
                 size_t NFail, 
                 bool useobsweight)
{
  double numerator = 0;
  double denominator = 0;
  double tempscore = -1;
  
  if (NFail == 0)
    return tempscore;
  
  double unbias;
    
  if (useobsweight)
    unbias = 0;
  else
    unbias = 1;
    
    
  // calculate the logrank for this split
  LeftW -= Left_Count_Censor[0];
  W -= Left_Count_Censor[0] + Right_Count_Censor[0];    
  
  for (size_t j = 1; j <= NFail and W > SurvWeightTH; j++)
  {
    numerator += LeftW*(Left_Count_Fail[j] + Right_Count_Fail[j])/W - Left_Count_Fail[j];
    denominator += LeftW*(Left_Count_Fail[j] + Right_Count_Fail[j])/W*(1- LeftW/W)*(W - Left_Count_Fail[j] - Right_Count_Fail[j])/(W - unbias);
    
    LeftW -= Left_Count_Fail[j] + Left_Count_Censor[j];
    W -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];
  }
  
  if (denominator > 0)
    tempscore = numerator*numerator/denominator;
  
  return tempscore;
}


double suplogrank_w(vec& Left_Count_Fail,
                    vec& Left_Count_Censor,
                    vec& Right_Count_Fail,
                    vec& Right_Count_Censor,
                    double LeftW,
                    double W,
                    size_t NFail,
                    bool useobsweight)
{
  Rcout << "      --- suplogrank weighted not implemented yet " << std::endl;
  
}
