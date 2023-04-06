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
      
      // sort the categories based on the probability of class j
      sort(cat_reduced.begin(), cat_reduced.begin() + true_cat, cat_class_compare);
      
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
        best_cat = record_cat_split(cat_reduced, temp_cat, true_cat, ncat);;
        best_score = temp_score;
      }
    }
  }else{
    // best split 
    
    if (true_cat > 10) // maxrun = 511
    {
      // first generate a random order of the categories 
      for (size_t j = 1; j < true_cat; j++)
        cat_reduced[j].score = rngl.rand_01();
      
      // sort the categories based on the probability of class j
      sort(cat_reduced.begin(), cat_reduced.begin() + true_cat, cat_class_compare);
      
      // calculate the score for first 511 = 2^(10-1)-1 types
      cla_uni_cat_score_best_large(cat_reduced, true_cat, ncat, nmin, 
                                   best_cat, best_score, rngl);
    }else{
      cla_uni_cat_score_best(cat_reduced, true_cat, ncat, nmin, 
                             best_cat, best_score, rngl);
    }
  }
  
  if (best_score > TempSplit.score)
  {
    TempSplit.value = best_cat;
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
  
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);

  for (size_t i = 0; i <= temp_cat; i++)
  {
    LeftSum += cat_reduced[i].Prob;
    Left_w += cat_reduced[i].weight;
  }
  
  for (size_t i = temp_cat+1; i < true_cat; i++)
  {
    RightSum += cat_reduced[i].Prob;
    Right_w += cat_reduced[i].weight;
  }
  
  if (Left_w > 0 && Right_w > 0)
  {
    return accu( square(LeftSum) ) / Left_w + 
           accu( square(RightSum) ) / Right_w;
  }

  return -1;
}

// best split 
void cla_uni_cat_score_best(std::vector<Cla_Cat_Class>& cat_reduced,
                            size_t true_cat,
                            size_t ncat,
                            size_t nmin,
                            size_t& best_cat,
                            double& best_score,
                            Rand& rngl)
{
  // this is not the real goright indicator, it only has true_cat
  // and the first element is used
  uvec temp_go(true_cat, fill::zeros);
  
  // set a maximum number of tries, this can deal with 11 categories completely
  // otherwise, do a random order and try 1024 of them
  
  size_t maxrun = (size_t) std::pow(2, true_cat - 1) - 1;
  
  if ( maxrun > 1024 )
  {
    // first generate a random order of the categories 
    for (size_t j = 1; j < true_cat; j++)
      cat_reduced[j].score = rngl.rand_01();
    
    // sort the categories based on the probability of class j
    sort(cat_reduced.begin(), cat_reduced.begin() + true_cat, cat_class_compare);
    
    //
    maxrun = 1024;
  }
  
  double Left_w = 0;
  double Right_w = 0;
  size_t Left_n = 0;
  size_t Right_n = 0;
  
  size_t nclass = cat_reduced[0].Prob.n_elem;
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);
  
  double tempscore;
  uvec best_go;
  
  for (size_t k = 0; k < maxrun; k++)
  {
    // add one to first cat, then adjust the binary indicators
    goright_roll_one(temp_go);

    // check sample size 
    Left_n = 0;
    Right_n = 0;
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j))
      {
        Right_n += cat_reduced[j].count;
      }else{
        Left_n += cat_reduced[j].count;
      }
    }
    
    // check nmin
    if (Left_n < nmin or Right_n < nmin)
      continue;
    
    // calculate scores
    Left_w = 0;
    Right_w = 0;
    LeftSum.zeros();
    RightSum.zeros();
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j))
      {
        RightSum += cat_reduced[j].Prob;
        Right_w += cat_reduced[j].weight;
      }else{
        LeftSum += cat_reduced[j].Prob;
        Left_w += cat_reduced[j].weight;
      }
    }
    
    tempscore = accu( square(LeftSum) ) / Left_w + 
                accu( square(RightSum) ) / Right_w;    
      
    if (tempscore > best_score)
    {
      best_go = temp_go;
      best_score = tempscore;
    }
  }
  
  // convert best_go to best_cat
  if (best_score > 0)
  {
    uvec goright(ncat + 1, fill::zeros); 
    
    for (size_t j = 0; j < true_cat; j++)
      goright(cat_reduced[j].cat) = best_go(j);
    
    best_cat = pack(ncat + 1, goright);    
  }
}

// best split for a large number of categories
void cla_uni_cat_score_best_large(std::vector<Cla_Cat_Class>& cat_reduced,
                                  size_t true_cat,
                                  size_t ncat,
                                  size_t nmin,
                                  size_t& best_cat,
                                  double& best_score,
                                  Rand& rngl)
{
  // this is for dealing with large number of categories
  size_t maxrun = 511;
  
  // this is not the real goright indicator, it only has true_cat
  // and the first element is used
  uvec temp_go(true_cat, fill::zeros);
  uvec temp_go_old(true_cat, fill::zeros);
  size_t nclass = cat_reduced[0].Prob.n_elem;
  
  size_t Left_n = 0;
  size_t Right_n = 0;  
  double Left_w = 0;
  double Right_w = 0;
  vec LeftSum(nclass, fill::zeros);
  vec RightSum(nclass, fill::zeros);
  
  // initiate everything on the left
  for (size_t j = 0; j < true_cat; j++)
  {
    Left_n += cat_reduced[j].count;
    Left_w += cat_reduced[j].weight;
    LeftSum += cat_reduced[j].Prob;
  }
  
  double tempscore;
  uvec best_go;
  
  for (size_t k = 0; k < maxrun; k++)
  {
    // add one to first cat, then adjust the binary indicators
    goright_roll_one(temp_go);
    
    // check sample size 
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) > temp_go_old(j))
      {
        // this cat move to right
        Right_n += cat_reduced[j].count;
        Left_n -= cat_reduced[j].count;
      }else if (temp_go(j) < temp_go_old(j)) {
        // this cat move to left
        Right_n -= cat_reduced[j].count;
        Left_n += cat_reduced[j].count;
      }
    }

    // check nmin
    if (Left_n < nmin or Right_n < nmin)
      continue;
    
    // move values based on change
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) > temp_go_old(j))
      {
        // this cat move to right
        Right_w += cat_reduced[j].weight;
        Left_w -= cat_reduced[j].weight;
        
        RightSum += cat_reduced[j].Prob;
        LeftSum -= cat_reduced[j].Prob;
        
      }else if (temp_go(j) < temp_go_old(j)){
        
        // this cat move to left
        Right_w -= cat_reduced[j].weight;
        Left_w += cat_reduced[j].weight;
        
        RightSum -= cat_reduced[j].Prob;
        LeftSum += cat_reduced[j].Prob;
      }
    }
    
    tempscore = accu( square(LeftSum) ) / Left_w + 
                accu( square(RightSum) ) / Right_w;    
    
    temp_go_old = temp_go;
    
    if (tempscore > best_score)
    {
      best_go = temp_go;
      best_score = tempscore;
    }
  }
  
  // convert best_go to best_cat
  if (best_score > 0)
  {
    uvec goright(ncat + 1, fill::zeros); 
    
    for (size_t j = 0; j < true_cat; j++)
      goright(cat_reduced[j].cat) = best_go(j);
    
    best_cat = pack(ncat + 1, goright);    
  }
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

  
  