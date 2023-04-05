//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Cla_Uni_Split_Cat(Split_Class& TempSplit,
                       const uvec& obs_id,
                       const vec& x,
                       const size_t ncat,
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
  // record each observation 
  size_t N = obs_id.n_elem;
  size_t nmin = N*alpha < 1 ? 1 : N*alpha;

  // I will initiate ncat+1 categories since factor x come from R and starts from 1
  // the first category will be empty
  std::vector<Cla_Cat_Class> cat_reduced(ncat + 1);
  
  // initiate values, ignore 0
  for (size_t j = 1; j < cat_reduced.size(); j++)
  {
    cat_reduced[j].initiate(j, nclass);
  }
  

  
  if (useobsweight){
    for (size_t i = 0; i < N; i++)
    {
      size_t temp_id = obs_id(i);
      size_t temp_cat = (size_t) x(temp_id);
      cat_reduced[temp_cat].Prob(Y(temp_id)) += obs_weight(temp_id);
      cat_reduced[temp_cat].count++;
      cat_reduced[temp_cat].weight += obs_weight(temp_id);
    }
  }else{
    for (size_t i = 0; i < N; i++)
    {
      size_t temp_cat = (size_t) x(obs_id(i));
      cat_reduced[temp_cat].Prob(Y(obs_id(i)))++;
      cat_reduced[temp_cat].count ++;
      cat_reduced[temp_cat].weight ++;
    }
  }
  
  // calculate other things
  size_t true_cat = 0;  
  
  for (size_t j = 1; j < cat_reduced.size(); j++)
  {
    if (cat_reduced[j].count)
    {
      true_cat++; // nonempty category
      cat_reduced[j].score = 1; // for sorting (random split)
    }
  }

  if (true_cat <= 1)
    return;

  // reorder them, nonempty categories comes first
  sort(cat_reduced.begin(), cat_reduced.end(), cat_class_compare);
  
  // used for recording
  size_t best_cat;
  double temp_score = 0;  
  double best_score = -1;
  
  // start split 
  if ( split_gen == 1 or split_gen == 2 )
  {
    
    for ( size_t k = 0; k < nsplit; k++ )
    {
      // first generate a random order of the categories
      for (size_t j = 1; j < true_cat; j++)
        cat_reduced[j].score = rngl.rand_01();
      
      // sort the categories 
      sort(cat_reduced.begin(), cat_reduced.begin()+true_cat, cat_class_compare);
      
      // rank split, figure out low and high index
      size_t lowindex = 0;
      size_t highindex = true_cat - 2;
      
      // if alpha > 0, this will (try to) force nmin for each child node
      if (alpha > 0)
        move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);
      
      // generate a random cut to split the categories
      size_t temp_cat = rngl.rand_sizet( lowindex, highindex);

      // this calculation is the same for weighted or unweighted
      temp_score = cla_uni_cat_score_cut(cat_reduced, temp_cat, true_cat);
      
      if (temp_score > best_score)
      {
        best_cat = temp_cat;
        best_score = temp_score;
      }
    }
  }
  
  if (best_score > TempSplit.score)
  {
    TempSplit.value = record_cat_split(cat_reduced, best_cat, true_cat, ncat);
    TempSplit.score = best_score;
  }
}


// this function is the same for weighted or unweighted
double cla_uni_cat_score_cut(std::vector<Cla_Cat_Class>& cat_reduced, 
                             size_t temp_cat, 
                             size_t true_cat)
{
  double Left_w = 0;
  double Right_w = 0;
  
  size_t nclass = cat_reduced[0].Prob.n_elem;
  
  vec LeftProb(nclass, fill::zeros);
  vec RightProb(nclass, fill::zeros);

  for (size_t i = 0; i <= temp_cat; i++)
  {
    LeftProb += cat_reduced[i].Prob;
    Left_w += cat_reduced[i].weight;
  }
  
  for (size_t i = temp_cat+1; i < true_cat; i++)
  {
    RightProb += cat_reduced[i].Prob;
    Right_w += cat_reduced[i].weight;
  }
  
  if (Left_w > 0 && Right_w > 0)
  {
    LeftProb /= Left_w;
    RightProb /= Right_w;
    
    return sum( square(LeftProb) ) * Left_w + sum( square(RightProb) )*Right_w;    
  }

  return -1;
}


// move categories
void move_cat_index(size_t& lowindex, 
                    size_t& highindex, 
                    std::vector<Cla_Cat_Class>& cat_reduced, 
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
double record_cat_split(std::vector<Cla_Cat_Class>& cat_reduced,
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

  
  