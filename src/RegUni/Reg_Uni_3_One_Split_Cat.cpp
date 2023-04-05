//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Split_Cat(Split_Class& TempSplit,
                       const uvec& obs_id,
                       const vec& x,
                       const size_t ncat,
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
  // the first element (category) of cat_reduced will always be set to 0 --- go left, 
  // but this category does not exist since the input data from R for factors only 
  // start from 1. 
  std::vector<Reg_Cat_Class> cat_reduced(ncat + 1);

  // record observations into the summary
  if (useobsweight){
    for (size_t i = 0; i < obs_id.size(); i++)
    {
      size_t temp_id = obs_id(i);
      size_t temp_cat = (size_t) x(temp_id);
      cat_reduced[temp_cat].y += Y(temp_id)*obs_weight(temp_id);   
      cat_reduced[temp_cat].count ++;      
      cat_reduced[temp_cat].weight += obs_weight(temp_id);
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
  
  // how many true nonempty categories
  size_t true_cat = 0;
  for (size_t j = 0; j < cat_reduced.size(); j++)
  {
    cat_reduced[j].cat = j;
    
    if (cat_reduced[j].count) 
    {
      true_cat++;
      cat_reduced[j].calculate_score(); // mean of each category
    }
  }
  
  // stop if only one category
  if (true_cat <= 1)
    return;  
  
  // this will move the 0 categories to the tail
  // nonempty categories are sorted based on mean of y
  sort(cat_reduced.begin(), cat_reduced.end(), cat_class_compare);
  
  
  ////////////////////////////////
  // start to find splitting rules
  ////////////////////////////////
  
  size_t best_cat;
  double temp_score = 0;  
  double best_score = -1;  

  // rank split, figure out low and high index
  size_t lowindex = 0;
  size_t highindex = true_cat - 2;
  
  // this will force nmin for each child node (below lowindex and above highindex)
  if (alpha > 0)
  {
    size_t N = obs_id.n_elem;
    size_t nmin = N*alpha < 1 ? 1 : N*alpha;
    
    // this only uses the count not weight
    move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);
  }

  // start split 
  
  if ( split_gen == 1 or split_gen == 2 )
  {
    for ( size_t k = 0; k < nsplit; k++ )
    {
      size_t temp_cat = rngl.rand_sizet(lowindex, highindex);
      
      // weighted or unweighted calculation is the same 
      // since we pre-calculated weights
      temp_score = reg_uni_cat_score_cut(cat_reduced, temp_cat, true_cat);
      
      if (temp_score > best_score)
      {      
        best_cat = temp_cat;
        best_score = temp_score;
      }
    }
  }else{
    // best split
      reg_uni_cat_score_best(cat_reduced, lowindex, highindex,
                             true_cat, best_cat, best_score);
  }
  
  if (best_score > TempSplit.score)
  {
    
    TempSplit.value = record_cat_split(cat_reduced, best_cat, true_cat, ncat);
    
    TempSplit.score = best_score;
    
  }
}


// calculate splitting scores at random cut of categories
double reg_uni_cat_score_cut(std::vector<Reg_Cat_Class>& cat_reduced, 
                             size_t temp_cat,
                             size_t true_cat)
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

// calculate best splitting scores
void reg_uni_cat_score_best(std::vector<Reg_Cat_Class>& cat_reduced, 
                            size_t lowindex, 
                            size_t highindex, 
                            size_t true_cat, 
                            size_t& best_cat, 
                            double& best_score)
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


// move categories
void move_cat_index(size_t& lowindex, 
                    size_t& highindex, 
                    std::vector<Reg_Cat_Class>& cat_reduced, 
                    size_t true_cat, 
                    size_t nmin)
{
  // Create a vector of pointers to Cat_Class
  std::vector<Cat_Class*> Categories(true_cat);
  
  for (size_t i = 0; i < true_cat; ++i) {
    Categories[i] = &cat_reduced[i];
  }
  
  // Call the function with the vector of Cat_Class pointers
  move_cat_index(lowindex, highindex, Categories, true_cat, nmin);
}

// record split
double record_cat_split(std::vector<Reg_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat)
{
  // the first element (category) of goright will always be set to 0 --- go left, 
  // but this category does not exist.
  uvec goright(ncat + 1, fill::zeros); 
  
  for (size_t i = 0; i <= best_cat; i++)
    goright[cat_reduced[i].cat] = 0;
  
  for (size_t i = best_cat + 1; i < true_cat; i++)
    goright[cat_reduced[i].cat] = 1;
  
  for (size_t i = true_cat + 1; i < ncat + 1; i++)
    goright[cat_reduced[i].cat] = 0; // for empty category, assign randomly
  
  return pack(ncat + 1, goright);
}

