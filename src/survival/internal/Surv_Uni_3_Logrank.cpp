//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;


// logrank split scores
void Surv_Uni_Logrank_Cont(Split_Class& TempSplit,
                           const uvec& obs_id,
                           const vec& x,
                           const uvec& Y, // Y is collapsed
                           const uvec& Censor, // Censor is collapsed
                           const size_t NFail,
                           const uvec& All_Fail,
                           const uvec& All_Risk, // cumulative
                           int nsplit,
                           double alpha,
                           Rand& rngl,
                           bool useobsweight,
                           const vec& obs_weight,
                           const vec& All_WFail,
                           const vec& All_WRisk,
                           const vec& All_W2Risk)
{
  size_t N = obs_id.n_elem;
  size_t temp_ind;  
  double temp_cut;
  double temp_score = -1;
  
  if (useobsweight)
  {
    // ---------- Weighted logrank path ----------
    if (nsplit > 0) // random split
    {
      for (int k = 0; k < nsplit; k++)
      {
        temp_ind = rngl.rand_sizet(0,N-1);
        temp_cut = x(obs_id( temp_ind ));
        
        temp_score = logrank_w_at_x_cut(obs_id, x, Y, Censor, NFail, 
                                         All_Fail, All_Risk,
                                         All_WFail, All_WRisk, All_W2Risk,
                                         obs_weight, temp_cut);
        
        if (temp_score > TempSplit.score)
        {
          TempSplit.value = temp_cut;
          TempSplit.score = temp_score;
        }
      }
      
      return;
    }
    
    uvec indices = stable_sort_index(x(obs_id));
    uvec obs_id_sorted = obs_id(indices);
    
    if ( x(obs_id_sorted(0)) == x(obs_id_sorted(N-1)) ) return;  
    
    size_t lowindex = 0;
    size_t highindex = N - 2;
    
    if (alpha > 0.5) alpha = 0.5;
    if (alpha > 0)
    {
      size_t nmin = (size_t) N*alpha;
      if (nmin < 1) nmin = 1;
      lowindex = nmin-1;
      highindex = N - nmin - 1;
    }
    
    if ( x(obs_id_sorted(lowindex)) == x(obs_id_sorted(lowindex+1)) or 
           x(obs_id_sorted(highindex)) == x(obs_id_sorted(highindex+1)) )
    {
      check_cont_index_sub(lowindex, highindex, x, obs_id_sorted);
      if (lowindex > highindex) return;
    }
    
    
    if (nsplit == 0) // best split  
    {
      logrank_w_best(indices, obs_id_sorted, x, Y, Censor, NFail,
                     All_Fail, All_Risk,
                     All_WFail, All_WRisk, All_W2Risk,
                     obs_weight, lowindex, highindex, 
                     TempSplit.value, TempSplit.score);
      return;
    }
    
  }else{
    // ---------- Unweighted logrank path (original) ----------
    if (nsplit > 0) // random split
    {
      for (int k = 0; k < nsplit; k++)
      {
        // generate a random cut off
        temp_ind = rngl.rand_sizet(0,N-1);
        temp_cut = x(obs_id( temp_ind ));
        
        // calculate logrank score
        temp_score = logrank_at_x_cut(obs_id, x, Y, Censor, NFail, 
                                      All_Fail, All_Risk, temp_cut);
        
        if (temp_score > TempSplit.score)
        {
          TempSplit.value = temp_cut;
          TempSplit.score = temp_score;
        }
      }
      
      return;
    }
    
    // this is the index used for Y and Censor (sorted based on x)
    uvec indices = stable_sort_index(x(obs_id));
    
    // this is the sorted obs_id for x 
    uvec obs_id_sorted = obs_id(indices);
    
    // check identical 
    if ( x(obs_id_sorted(0)) == x(obs_id_sorted(N-1)) ) return;  
    
    // set low and high index
    size_t lowindex = 0; // less equal goes to left
    size_t highindex = N - 2;
    
    // alpha is only effective when x can be sorted
    // this will try to force nmin for each child node
    if (alpha > 0.5) alpha = 0.5;
    if (alpha > 0)
    {
      size_t nmin = (size_t) N*alpha;
      if (nmin < 1) nmin = 1;
      
      lowindex = nmin-1; // less equal goes to left
      highindex = N - nmin - 1;
    }
    
    // if ties, move index to better locations
    if ( x(obs_id_sorted(lowindex)) == x(obs_id_sorted(lowindex+1)) or 
           x(obs_id_sorted(highindex)) == x(obs_id_sorted(highindex+1)) )
    {
      check_cont_index_sub(lowindex, highindex, x, obs_id_sorted);
      
      if (lowindex > highindex)
      {
        return;
      }
    }
    
    
    
    if (nsplit == 0) // best split  
    {

      logrank_best(indices, obs_id_sorted, x, Y, Censor, NFail,
                   All_Fail, All_Risk, lowindex, highindex, 
                   TempSplit.value, TempSplit.score);
      return;
    }
    
  } // end unweighted path
  
}


// logrank score at a numerical cut of x value
double logrank_at_x_cut(const uvec& obs_id,
                        const vec& x,
                        const uvec& Y, //collapsed
                        const uvec& Censor, //collapsed
                        const size_t NFail,
                        const uvec& All_Fail,
                        const uvec& All_Risk,                        
                        double a_random_cut)
{
  size_t N = obs_id.n_elem;
  uvec Left_Risk(NFail+1, fill::zeros);
  uvec Left_Fail(NFail+1, fill::zeros);

  for (size_t i = 0; i<N; i++)
  {
    //If x is less than the random cut, go left
    if (x(obs_id(i)) <= a_random_cut)
    {
      Left_Risk(Y(i)) ++;
      Left_Fail(Y(i)) += Censor(i);
    }
  }
  
  // cumulative at risk counts for left
  cumsum_rev(Left_Risk);
  
  return logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);

}

// logrank score at a random index number, provided with sorted index
double logrank_at_id_index(const uvec& indices, // index for Y, sorted by x
                           const uvec& Y, //collapsed
                           const uvec& Censor, //collapsed
                           const size_t NFail,
                           const uvec& All_Fail, 
                           const uvec& All_Risk,
                           size_t a_random_ind)
{
  uvec Left_Risk(NFail+1, fill::zeros);
  uvec Left_Fail(NFail+1, fill::zeros);
  
  for (size_t i = 0; i <= a_random_ind; i++)
  {
    Left_Risk(Y(indices(i))) ++;
    Left_Fail(Y(indices(i))) += Censor(indices(i));
  }
  
  // cumulative at risk counts for left
  cumsum_rev(Left_Risk);
  
  return logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
}


// logrank best score 
void logrank_best(const uvec& indices, // index for Y, sorted by x
                  const uvec& obs_id_sorted, // index for x, sorted by x
                  const vec& x, 
                  const uvec& Y, //collapsed
                  const uvec& Censor, //collapsed
                  const size_t NFail,
                  const uvec& All_Fail, 
                  const uvec& All_Risk, 
                  size_t lowindex,
                  size_t highindex,
                  double& temp_cut, 
                  double& temp_score)
{

  double score = -1;
  uvec Left_Risk(NFail+1, fill::zeros);
  uvec Left_Fail(NFail+1, fill::zeros);
  uvec Left_Risk_Cum;
  
  // initiate the failure and censoring counts
  for (size_t i = 0; i< lowindex; i++)
  {
    Left_Risk(Y(indices(i))) ++;
    Left_Fail(Y(indices(i))) += Censor(indices(i));
  }
  
  for (size_t i = lowindex; i <= highindex; i++)
  {
    // move things to the left node
    Left_Risk(Y(indices(i))) ++;
    Left_Fail(Y(indices(i))) += Censor(indices(i));    
    
    if(x(obs_id_sorted(i)) < x(obs_id_sorted(i+1))) // if not a tie location
    {
      // calculate cumulative risk of left
      Left_Risk_Cum = Left_Risk;
      cumsum_rev(Left_Risk_Cum);
      
      // get scores
      if (Left_Risk_Cum(0) == 0 or Left_Risk_Cum(0) == All_Risk(0))
        score = -1;
      else
        score = logrank(Left_Fail, Left_Risk_Cum, All_Fail, All_Risk);
      
      if (score > temp_score)
      {
        temp_cut = (x(obs_id_sorted(i)) + x(obs_id_sorted(i + 1)))/2 ;
        temp_score = score;
      }
    }
  }
}


// logrank score given pre-processed vectors
double logrank(const uvec& Left_Fail, 
               const uvec& Left_Risk, // left Fail is already cumulative
               const uvec& All_Fail,
               const uvec& All_Risk) // also cumulative 
{

  // cumulative at risk counts for left
  size_t NFail = All_Risk.n_elem - 1;

  double Oj = 0, Eij = 0;
  double Nj = 0, Nij = 0;
  double Z = 0, V = 0;
  
  for (size_t j = 1; j < NFail; j++) // failure times start from 1
  {
    Oj = All_Fail(j);
    Nij = Left_Risk(j);
    Nj = All_Risk(j);
    Eij = Oj * Nij / Nj;
    Z += Left_Fail(j) - Eij;
    V += Eij * (1 - Oj / Nj) * (Nj - Nij) / (Nj - 1);
  }
  
  Oj = All_Fail(NFail);
  
  // last time point
  if (Oj > 1)
  {
    Nij = Left_Risk(NFail);
    Nj = All_Risk(NFail);
    Eij = Oj * Nij / Nj;
    Z += Left_Fail(NFail) - Eij;
    V += Eij * (1 - Oj / Nj) * (Nj - Nij) / (Nj - 1);
  }

  return Z*Z / V;
}

// Calculate logrank score at x value cut, sequential calculation without vector
// this is not validated yet, not currently used, maybe for later?
double logrank_at_x_cut_novec(const uvec& obs_id,
                              const vec& x,
                              const uvec& Y, //collapsed
                              const uvec& Censor, //collapsed
                              const size_t NFail,
                              double a_random_cut)
{
  size_t N = obs_id.size();
  
  size_t N_L = 0, N_All = 0; // at risk counts
  size_t O_L = 0, O_All = 0; // fail counts
  double V = 0, Z = 0;
  
  size_t current_timepoint = Y(N-1);
  
  for (size_t i=N-1; i>0; i--)
  {
    // starting from the last time point
    if (Y(i) < current_timepoint and N_All > 1)
    {
      // finish scores of the previous (larger) time points
      double E = (double)O_L*N_L/N_All;
      Z += O_L - E;
      V += E*(N_All - O_All)/N_All*(N_All - N_L)/(N_All-1);
      
      // reset failure counts
      O_L = 0;
      O_All = 0;
    }
    
    //
    N_All++;
    O_All+= Censor(i);
    
    // 
    if ( x(obs_id(i)) <= a_random_cut ) // go left
    {
      N_L++;
      O_L += Censor(i);
    }
    
    if (i == 0) // conclude the calculation
    {
      double E = (double)O_L*N_L/N_All;
      Z += O_L - E;
      V += E*(N_All - O_All)/N_All*(N_All - N_L)/(N_All-1);      
      
      break;
    }
  }
  
  if (V > 0)
    return Z*Z / V;
  else
    return -1;
}


/////////////////////////////////////////////
// logrank split for categorical variables //
/////////////////////////////////////////////

void Surv_Uni_Logrank_Cat(Split_Class& TempSplit,
                          const uvec& obs_id,
                          const vec& x,
                          const size_t ncat,
                          const uvec& Y, // Y is collapsed
                          const uvec& Censor, // Censor is collapsed
                          const size_t NFail,
                          const uvec& All_Fail,
                          const uvec& All_Risk, // cumulative
                          int nsplit,
                          double alpha,
                          Rand& rngl)
{
  // record each observation 
  size_t N = obs_id.n_elem;
  size_t nmin = N*alpha < 1 ? 1 : N*alpha;

  // I will initiate ncat+1 categories since factor x come from R and starts from 1
  // the first category will be empty
  std::vector<Surv_Cat_Class> cat_reduced(ncat + 1);
  
  // initiate values, ignore 0
  for (size_t j = 1; j < cat_reduced.size(); j++)
  {
    cat_reduced[j].initiate(j, NFail);
  }

  // copy data into reduced list 
  for (size_t i = 0; i < N; i++)
  {
    size_t temp_cat = (size_t) x(obs_id(i));
    
    cat_reduced[temp_cat].count++;
    cat_reduced[temp_cat].weight++;
    
    if (Censor(i)) // failure obs
    {
      cat_reduced[temp_cat].RiskCount(Y(i))++;
      cat_reduced[temp_cat].FailCount(Y(i))++;
      cat_reduced[temp_cat].nfail++;
    }else{
      cat_reduced[temp_cat].RiskCount(Y(i))++;
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
      cumsum_rev(cat_reduced[j].RiskCount);
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
  if (nsplit > 0)
  {
    for ( int k = 0; k < nsplit; k++ )
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
      size_t temp_cat = rngl.rand_sizet( lowindex, highindex );
      
      // copy things into fail and risk vectors
      uvec Left_Fail(NFail+1, fill::zeros);
      uvec Left_Risk(NFail+1, fill::zeros);

      for (size_t i = 0; i <= temp_cat; i++)
      {
        Left_Fail += cat_reduced[i].FailCount;
        Left_Risk += cat_reduced[i].RiskCount;
      }
      
      // cumulative at risk counts for left
      // cumsum_rev(Left_Risk);

      temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);      

      if (temp_score > best_score)
      {
        best_cat = record_cat_split(cat_reduced, temp_cat, true_cat, ncat);
        best_score = temp_score;
      }
    }
  }else{
    // best split
    if (true_cat > 6) // random 63 = 2^6-1 binary partitions
    {
      size_t maxrun = 63;
      size_t tlength = All_Fail.n_elem;
      uvec Left_Fail(tlength, fill::zeros);
      uvec Left_Risk(tlength, fill::zeros);
      double tempscore;
      uvec best_go;
      
      for (size_t k = 0; k < maxrun; k++)
      {
        uvec temp_go(true_cat, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          temp_go(j) = rngl.rand_01() < 0.5 ? 1 : 0;
        
        if (temp_go.min() == temp_go.max()) continue;
        
        size_t Left_n = 0;
        for (size_t j = 0; j < true_cat; j++)
          if (temp_go(j) == 0) Left_n += cat_reduced[j].count;
        if (Left_n < nmin || (N - Left_n) < nmin) continue;
        
        Left_Fail.zeros();
        Left_Risk.zeros();
        for (size_t j = 0; j < true_cat; j++)
        {
          if (temp_go(j) == 1)
          {
            Left_Fail += cat_reduced[j].FailCount;
            Left_Risk += cat_reduced[j].RiskCount;
          }
        }
        
        tempscore = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
        
        if (tempscore > best_score)
        {
          best_go = temp_go;
          best_score = tempscore;
        }
      }
      
      if (best_score > 0)
      {
        uvec goright(ncat + 1, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          goright(cat_reduced[j].cat) = best_go(j);
        best_cat = pack(ncat + 1, goright);
      }
    }else{
      logrank_best_cat(cat_reduced, true_cat, ncat, nmin, N,
                       All_Fail, All_Risk, 
                       best_cat, best_score, rngl);
    }
  }
  
  if (best_score > TempSplit.score)
  {
    TempSplit.value = best_cat;
    TempSplit.score = best_score;
  }

  
}





// best split this will go through all combinations
void logrank_best_cat(std::vector<Surv_Cat_Class>& cat_reduced,
                      size_t true_cat,
                      size_t ncat,
                      size_t nmin,
                      size_t N,
                      const uvec& All_Fail,
                      const uvec& All_Risk, // cumulative
                      size_t& best_cat,
                      double& best_score,
                      Rand& rngl)
{
  // this is not the real goright indicator, it only has true_cat
  // and the first element is used
  uvec temp_go(true_cat, fill::zeros);
  
  // number of all possible combinations
  size_t maxrun = (size_t) std::pow(2, true_cat - 1) - 1;
  
  // prepare for tests
  size_t tlength = All_Fail.n_elem;
  size_t Left_n = 0;
  uvec Left_Fail(tlength, fill::zeros);
  uvec Left_Risk(tlength, fill::zeros);
  
  double tempscore;
  uvec best_go;
  
  for (size_t k = 0; k < maxrun; k++)
  {
    // add one to first cat, then adjust the binary indicators
    goright_roll_one(temp_go);
    
    // check sample size 
    Left_n = 0;
    Left_Fail.zeros();
    Left_Risk.zeros();
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) == 0)
        Left_n += cat_reduced[j].count;
    }
    
    // check nmin
    if (Left_n < nmin or (N - Left_n) < nmin)
      continue;
    
    // calculate scores
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) == 1)
      {
        Left_Fail += cat_reduced[j].FailCount;
        Left_Risk += cat_reduced[j].RiskCount;
      }
      
      // cumsum_rev(Left_Risk);
    }

    tempscore = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
    
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

///////////////////////////////////////////////////
// Weighted logrank categorical best split        //
///////////////////////////////////////////////////

void logrank_w_best_cat(std::vector<Surv_Cat_Class>& cat_reduced,
                        size_t true_cat,
                        size_t ncat,
                        size_t nmin,
                        size_t N,
                        const uvec& All_Fail,
                        const uvec& All_Risk,
                        const vec& All_WFail,
                        const vec& All_WRisk,
                        const vec& All_W2Risk,
                        size_t& best_cat,
                        double& best_score,
                        Rand& rngl)
{
  uvec temp_go(true_cat, fill::zeros);
  size_t maxrun = (size_t) std::pow(2, true_cat - 1) - 1;
  
  size_t tlength = All_Fail.n_elem;
  size_t Left_n = 0;
  vec Left_WFail(tlength, fill::zeros);
  vec Left_WRisk(tlength, fill::zeros);
  vec Left_W2Risk(tlength, fill::zeros);
  
  double tempscore;
  uvec best_go;
  
  for (size_t k = 0; k < maxrun; k++)
  {
    goright_roll_one(temp_go);
    
    Left_n = 0;
    Left_WFail.zeros();
    Left_WRisk.zeros();
    Left_W2Risk.zeros();
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) == 0)
        Left_n += cat_reduced[j].count;
    }
    
    if (Left_n < nmin or (N - Left_n) < nmin)
      continue;
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) == 1)
      {
        Left_WFail += cat_reduced[j].WFailCount;
        Left_WRisk += cat_reduced[j].WRiskCount;
        Left_W2Risk += cat_reduced[j].W2RiskCount;
      }
    }
    
    tempscore = logrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                          All_WFail, All_WRisk, All_W2Risk,
                          All_Fail, All_Risk);
    
    if (tempscore > best_score)
    {
      best_go = temp_go;
      best_score = tempscore;
    }
  }
  
  if (best_score > 0)
  {
    uvec goright(ncat + 1, fill::zeros);
    for (size_t j = 0; j < true_cat; j++)
      goright(cat_reduced[j].cat) = best_go(j);
    best_cat = pack(ncat + 1, goright);
  }
}

///////////////////////////////////////////////////
// Weighted logrank categorical split             //
///////////////////////////////////////////////////

void Surv_Uni_Logrank_Cat_W(Split_Class& TempSplit,
                            const uvec& obs_id,
                            const vec& x,
                            const size_t ncat,
                            const uvec& Y,
                            const uvec& Censor,
                            const size_t NFail,
                            const uvec& All_Fail,
                            const uvec& All_Risk,
                            const vec& All_WFail,
                            const vec& All_WRisk,
                            const vec& All_W2Risk,
                            const vec& obs_weight,
                            int nsplit,
                            double alpha,
                            Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = N*alpha < 1 ? 1 : N*alpha;
  
  std::vector<Surv_Cat_Class> cat_reduced(ncat + 1);
  
  for (size_t j = 1; j < cat_reduced.size(); j++)
    cat_reduced[j].initiate(j, NFail);
  
  // Accumulate both unweighted and weighted counts per category
  for (size_t i = 0; i < N; i++)
  {
    size_t temp_cat = (size_t) x(obs_id(i));
    double wi = obs_weight(i);
    double w2i = wi * wi;
    
    cat_reduced[temp_cat].count++;
    cat_reduced[temp_cat].weight++;
    
    if (Censor(i))
    {
      cat_reduced[temp_cat].RiskCount(Y(i))++;
      cat_reduced[temp_cat].FailCount(Y(i))++;
      cat_reduced[temp_cat].WRiskCount(Y(i)) += wi;
      cat_reduced[temp_cat].WFailCount(Y(i)) += wi;
      cat_reduced[temp_cat].W2RiskCount(Y(i)) += w2i;
      cat_reduced[temp_cat].nfail++;
    }else{
      cat_reduced[temp_cat].RiskCount(Y(i))++;
      cat_reduced[temp_cat].WRiskCount(Y(i)) += wi;
      cat_reduced[temp_cat].W2RiskCount(Y(i)) += w2i;
    }
  }
  
  size_t true_cat = 0;
  
  for (size_t j = 1; j < cat_reduced.size(); j++)
  {
    if (cat_reduced[j].count)
    {
      true_cat++;
      cat_reduced[j].score = 1;
      cumsum_rev(cat_reduced[j].RiskCount);
      cumsum_rev(cat_reduced[j].WRiskCount);
      cumsum_rev(cat_reduced[j].W2RiskCount);
    }
  }
  
  if (true_cat <= 1)
    return;
  
  sort(cat_reduced.begin(), cat_reduced.end(), cat_class_compare);
  
  size_t best_cat;
  double temp_score = 0;
  double best_score = -1;
  
  if (nsplit > 0)
  {
    for (int k = 0; k < nsplit; k++)
    {
      for (size_t j = 1; j < true_cat; j++)
        cat_reduced[j].score = rngl.rand_01();
      
      sort(cat_reduced.begin(), cat_reduced.begin() + true_cat, cat_class_compare);
      
      size_t lowindex = 0;
      size_t highindex = true_cat - 2;
      
      if (alpha > 0)
        move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);
      
      size_t temp_cat_ind = rngl.rand_sizet(lowindex, highindex);
      
      vec Left_WFail(NFail+1, fill::zeros);
      vec Left_WRisk(NFail+1, fill::zeros);
      vec Left_W2Risk(NFail+1, fill::zeros);
      
      for (size_t i = 0; i <= temp_cat_ind; i++)
      {
        Left_WFail += cat_reduced[i].WFailCount;
        Left_WRisk += cat_reduced[i].WRiskCount;
        Left_W2Risk += cat_reduced[i].W2RiskCount;
      }
      
      temp_score = logrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                             All_WFail, All_WRisk, All_W2Risk,
                             All_Fail, All_Risk);
      
      if (temp_score > best_score)
      {
        best_cat = record_cat_split(cat_reduced, temp_cat_ind, true_cat, ncat);
        best_score = temp_score;
      }
    }
  }else{
    if (true_cat > 6) // random 63 binary partitions
    {
      size_t maxrun = 63;
      size_t tlength = All_Fail.n_elem;
      vec Left_WFail(tlength, fill::zeros);
      vec Left_WRisk(tlength, fill::zeros);
      vec Left_W2Risk(tlength, fill::zeros);
      double tempscore;
      uvec best_go;
      
      for (size_t k = 0; k < maxrun; k++)
      {
        uvec temp_go(true_cat, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          temp_go(j) = rngl.rand_01() < 0.5 ? 1 : 0;
        
        if (temp_go.min() == temp_go.max()) continue;
        
        size_t Left_n = 0;
        for (size_t j = 0; j < true_cat; j++)
          if (temp_go(j) == 0) Left_n += cat_reduced[j].count;
        if (Left_n < nmin || (N - Left_n) < nmin) continue;
        
        Left_WFail.zeros();
        Left_WRisk.zeros();
        Left_W2Risk.zeros();
        for (size_t j = 0; j < true_cat; j++)
        {
          if (temp_go(j) == 1)
          {
            Left_WFail += cat_reduced[j].WFailCount;
            Left_WRisk += cat_reduced[j].WRiskCount;
            Left_W2Risk += cat_reduced[j].W2RiskCount;
          }
        }
        
        tempscore = logrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                               All_WFail, All_WRisk, All_W2Risk,
                               All_Fail, All_Risk);
        
        if (tempscore > best_score)
        {
          best_go = temp_go;
          best_score = tempscore;
        }
      }
      
      if (best_score > 0)
      {
        uvec goright(ncat + 1, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          goright(cat_reduced[j].cat) = best_go(j);
        best_cat = pack(ncat + 1, goright);
      }
    }else{
      logrank_w_best_cat(cat_reduced, true_cat, ncat, nmin, N,
                         All_Fail, All_Risk,
                         All_WFail, All_WRisk, All_W2Risk,
                         best_cat, best_score, rngl);
    }
  }
  
  if (best_score > TempSplit.score)
  {
    TempSplit.value = best_cat;
    TempSplit.score = best_score;
  }
}

///////////////////////////////////////////////////
// Suplogrank categorical functions               //
///////////////////////////////////////////////////

// Unweighted suplogrank best split for categories
void suplogrank_best_cat(std::vector<Surv_Cat_Class>& cat_reduced,
                         size_t true_cat,
                         size_t ncat,
                         size_t nmin,
                         size_t N,
                         const uvec& All_Fail,
                         const vec& All_Risk,
                         vec& Temp_Vec,
                         size_t& best_cat,
                         double& best_score,
                         Rand& rngl)
{
  uvec temp_go(true_cat, fill::zeros);
  size_t maxrun = (size_t) std::pow(2, true_cat - 1) - 1;
  
  size_t tlength = All_Fail.n_elem;
  size_t Left_n = 0;
  uvec Left_Fail(tlength, fill::zeros);
  uvec Left_Risk(tlength, fill::zeros);
  
  double tempscore;
  uvec best_go;
  
  for (size_t k = 0; k < maxrun; k++)
  {
    goright_roll_one(temp_go);
    
    Left_n = 0;
    Left_Fail.zeros();
    Left_Risk.zeros();
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) == 0)
        Left_n += cat_reduced[j].count;
    }
    
    if (Left_n < nmin or (N - Left_n) < nmin)
      continue;
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) == 1)
      {
        Left_Fail += cat_reduced[j].FailCount;
        Left_Risk += cat_reduced[j].RiskCount;
      }
    }
    
    tempscore = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
    
    if (tempscore > best_score)
    {
      best_go = temp_go;
      best_score = tempscore;
    }
  }
  
  if (best_score > 0)
  {
    uvec goright(ncat + 1, fill::zeros);
    for (size_t j = 0; j < true_cat; j++)
      goright(cat_reduced[j].cat) = best_go(j);
    best_cat = pack(ncat + 1, goright);
  }
}

// Weighted suplogrank best split for categories
void suplogrank_w_best_cat(std::vector<Surv_Cat_Class>& cat_reduced,
                           size_t true_cat,
                           size_t ncat,
                           size_t nmin,
                           size_t N,
                           const uvec& All_Fail,
                           const vec& All_Risk,
                           const vec& All_WFail,
                           const vec& All_WRisk,
                           const vec& All_W2Risk,
                           size_t& best_cat,
                           double& best_score,
                           Rand& rngl)
{
  uvec temp_go(true_cat, fill::zeros);
  size_t maxrun = (size_t) std::pow(2, true_cat - 1) - 1;
  
  size_t tlength = All_Fail.n_elem;
  size_t Left_n = 0;
  vec Left_WFail(tlength, fill::zeros);
  vec Left_WRisk(tlength, fill::zeros);
  vec Left_W2Risk(tlength, fill::zeros);
  
  double tempscore;
  uvec best_go;
  
  for (size_t k = 0; k < maxrun; k++)
  {
    goright_roll_one(temp_go);
    
    Left_n = 0;
    Left_WFail.zeros();
    Left_WRisk.zeros();
    Left_W2Risk.zeros();
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) == 0)
        Left_n += cat_reduced[j].count;
    }
    
    if (Left_n < nmin or (N - Left_n) < nmin)
      continue;
    
    for (size_t j = 0; j < true_cat; j++)
    {
      if (temp_go(j) == 1)
      {
        Left_WFail += cat_reduced[j].WFailCount;
        Left_WRisk += cat_reduced[j].WRiskCount;
        Left_W2Risk += cat_reduced[j].W2RiskCount;
      }
    }
    
    tempscore = suplogrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                             All_WFail, All_WRisk, All_W2Risk,
                             All_Fail, All_Risk);
    
    if (tempscore > best_score)
    {
      best_go = temp_go;
      best_score = tempscore;
    }
  }
  
  if (best_score > 0)
  {
    uvec goright(ncat + 1, fill::zeros);
    for (size_t j = 0; j < true_cat; j++)
      goright(cat_reduced[j].cat) = best_go(j);
    best_cat = pack(ncat + 1, goright);
  }
}

///////////////////////////////////////////////////
// Unweighted suplogrank categorical split        //
///////////////////////////////////////////////////

void Surv_Uni_SupLogrank_Cat(Split_Class& TempSplit,
                             const uvec& obs_id,
                             const vec& x,
                             const size_t ncat,
                             const uvec& Y,
                             const uvec& Censor,
                             const size_t NFail,
                             const uvec& All_Fail,
                             const vec& All_Risk,
                             int nsplit,
                             double alpha,
                             Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = N*alpha < 1 ? 1 : N*alpha;
  
  // Compute Temp_Vec for suplogrank (same as Find_A_Split)
  vec Temp_Vec(NFail+1, fill::zeros);
  uvec All_Fail_u = All_Fail; // already uvec but in case of type issues
  Temp_Vec = 1.0 - conv_to<vec>::from(All_Fail - 1.0)/(All_Risk - 1.0);
  Temp_Vec = Temp_Vec % conv_to<vec>::from(All_Fail)/All_Risk;
  for (size_t i = 0; i < All_Risk.n_elem; i++)
    if (All_Risk(i) < 2)
      Temp_Vec(i) = 0;
  
  std::vector<Surv_Cat_Class> cat_reduced(ncat + 1);
  
  for (size_t j = 1; j < cat_reduced.size(); j++)
    cat_reduced[j].initiate(j, NFail);
  
  for (size_t i = 0; i < N; i++)
  {
    size_t temp_cat = (size_t) x(obs_id(i));
    
    cat_reduced[temp_cat].count++;
    cat_reduced[temp_cat].weight++;
    
    if (Censor(i))
    {
      cat_reduced[temp_cat].RiskCount(Y(i))++;
      cat_reduced[temp_cat].FailCount(Y(i))++;
      cat_reduced[temp_cat].nfail++;
    }else{
      cat_reduced[temp_cat].RiskCount(Y(i))++;
    }
  }
  
  size_t true_cat = 0;
  
  for (size_t j = 1; j < cat_reduced.size(); j++)
  {
    if (cat_reduced[j].count)
    {
      true_cat++;
      cat_reduced[j].score = 1;
      cumsum_rev(cat_reduced[j].RiskCount);
    }
  }
  
  if (true_cat <= 1)
    return;
  
  sort(cat_reduced.begin(), cat_reduced.end(), cat_class_compare);
  
  size_t best_cat;
  double temp_score = 0;
  double best_score = -1;
  
  if (nsplit > 0)
  {
    for (int k = 0; k < nsplit; k++)
    {
      for (size_t j = 1; j < true_cat; j++)
        cat_reduced[j].score = rngl.rand_01();
      
      sort(cat_reduced.begin(), cat_reduced.begin() + true_cat, cat_class_compare);
      
      size_t lowindex = 0;
      size_t highindex = true_cat - 2;
      
      if (alpha > 0)
        move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);
      
      size_t temp_cat_ind = rngl.rand_sizet(lowindex, highindex);
      
      uvec Left_Fail(NFail+1, fill::zeros);
      uvec Left_Risk(NFail+1, fill::zeros);
      
      for (size_t i = 0; i <= temp_cat_ind; i++)
      {
        Left_Fail += cat_reduced[i].FailCount;
        Left_Risk += cat_reduced[i].RiskCount;
      }
      
      temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
      
      if (temp_score > best_score)
      {
        best_cat = record_cat_split(cat_reduced, temp_cat_ind, true_cat, ncat);
        best_score = temp_score;
      }
    }
  }else{
    if (true_cat > 6) // random 63 binary partitions
    {
      size_t maxrun = 63;
      size_t tlength = All_Fail.n_elem;
      uvec Left_Fail(tlength, fill::zeros);
      uvec Left_Risk(tlength, fill::zeros);
      double tempscore;
      uvec best_go;
      
      for (size_t k = 0; k < maxrun; k++)
      {
        uvec temp_go(true_cat, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          temp_go(j) = rngl.rand_01() < 0.5 ? 1 : 0;
        
        if (temp_go.min() == temp_go.max()) continue;
        
        size_t Left_n = 0;
        for (size_t j = 0; j < true_cat; j++)
          if (temp_go(j) == 0) Left_n += cat_reduced[j].count;
        if (Left_n < nmin || (N - Left_n) < nmin) continue;
        
        Left_Fail.zeros();
        Left_Risk.zeros();
        for (size_t j = 0; j < true_cat; j++)
        {
          if (temp_go(j) == 1)
          {
            Left_Fail += cat_reduced[j].FailCount;
            Left_Risk += cat_reduced[j].RiskCount;
          }
        }
        
        tempscore = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
        
        if (tempscore > best_score)
        {
          best_go = temp_go;
          best_score = tempscore;
        }
      }
      
      if (best_score > 0)
      {
        uvec goright(ncat + 1, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          goright(cat_reduced[j].cat) = best_go(j);
        best_cat = pack(ncat + 1, goright);
      }
    }else{
      suplogrank_best_cat(cat_reduced, true_cat, ncat, nmin, N,
                          All_Fail, All_Risk, Temp_Vec,
                          best_cat, best_score, rngl);
    }
  }
  
  if (best_score > TempSplit.score)
  {
    TempSplit.value = best_cat;
    TempSplit.score = best_score;
  }
}

///////////////////////////////////////////////////
// Weighted suplogrank categorical split           //
///////////////////////////////////////////////////

void Surv_Uni_SupLogrank_Cat_W(Split_Class& TempSplit,
                               const uvec& obs_id,
                               const vec& x,
                               const size_t ncat,
                               const uvec& Y,
                               const uvec& Censor,
                               const size_t NFail,
                               const uvec& All_Fail,
                               const vec& All_Risk,
                               const vec& All_WFail,
                               const vec& All_WRisk,
                               const vec& All_W2Risk,
                               const vec& obs_weight,
                               int nsplit,
                               double alpha,
                               Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = N*alpha < 1 ? 1 : N*alpha;
  
  std::vector<Surv_Cat_Class> cat_reduced(ncat + 1);
  
  for (size_t j = 1; j < cat_reduced.size(); j++)
    cat_reduced[j].initiate(j, NFail);
  
  for (size_t i = 0; i < N; i++)
  {
    size_t temp_cat = (size_t) x(obs_id(i));
    double wi = obs_weight(i);
    double w2i = wi * wi;
    
    cat_reduced[temp_cat].count++;
    cat_reduced[temp_cat].weight++;
    
    if (Censor(i))
    {
      cat_reduced[temp_cat].RiskCount(Y(i))++;
      cat_reduced[temp_cat].FailCount(Y(i))++;
      cat_reduced[temp_cat].WRiskCount(Y(i)) += wi;
      cat_reduced[temp_cat].WFailCount(Y(i)) += wi;
      cat_reduced[temp_cat].W2RiskCount(Y(i)) += w2i;
      cat_reduced[temp_cat].nfail++;
    }else{
      cat_reduced[temp_cat].RiskCount(Y(i))++;
      cat_reduced[temp_cat].WRiskCount(Y(i)) += wi;
      cat_reduced[temp_cat].W2RiskCount(Y(i)) += w2i;
    }
  }
  
  size_t true_cat = 0;
  
  for (size_t j = 1; j < cat_reduced.size(); j++)
  {
    if (cat_reduced[j].count)
    {
      true_cat++;
      cat_reduced[j].score = 1;
      cumsum_rev(cat_reduced[j].RiskCount);
      cumsum_rev(cat_reduced[j].WRiskCount);
      cumsum_rev(cat_reduced[j].W2RiskCount);
    }
  }
  
  if (true_cat <= 1)
    return;
  
  sort(cat_reduced.begin(), cat_reduced.end(), cat_class_compare);
  
  size_t best_cat;
  double temp_score = 0;
  double best_score = -1;
  
  if (nsplit > 0)
  {
    for (int k = 0; k < nsplit; k++)
    {
      for (size_t j = 1; j < true_cat; j++)
        cat_reduced[j].score = rngl.rand_01();
      
      sort(cat_reduced.begin(), cat_reduced.begin() + true_cat, cat_class_compare);
      
      size_t lowindex = 0;
      size_t highindex = true_cat - 2;
      
      if (alpha > 0)
        move_cat_index(lowindex, highindex, cat_reduced, true_cat, nmin);
      
      size_t temp_cat_ind = rngl.rand_sizet(lowindex, highindex);
      
      vec Left_WFail(NFail+1, fill::zeros);
      vec Left_WRisk(NFail+1, fill::zeros);
      vec Left_W2Risk(NFail+1, fill::zeros);
      
      for (size_t i = 0; i <= temp_cat_ind; i++)
      {
        Left_WFail += cat_reduced[i].WFailCount;
        Left_WRisk += cat_reduced[i].WRiskCount;
        Left_W2Risk += cat_reduced[i].W2RiskCount;
      }
      
      temp_score = suplogrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                                All_WFail, All_WRisk, All_W2Risk,
                                All_Fail, All_Risk);
      
      if (temp_score > best_score)
      {
        best_cat = record_cat_split(cat_reduced, temp_cat_ind, true_cat, ncat);
        best_score = temp_score;
      }
    }
  }else{
    if (true_cat > 6) // random 63 binary partitions
    {
      size_t maxrun = 63;
      size_t tlength = All_Fail.n_elem;
      vec Left_WFail(tlength, fill::zeros);
      vec Left_WRisk(tlength, fill::zeros);
      vec Left_W2Risk(tlength, fill::zeros);
      double tempscore;
      uvec best_go;
      
      for (size_t k = 0; k < maxrun; k++)
      {
        uvec temp_go(true_cat, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          temp_go(j) = rngl.rand_01() < 0.5 ? 1 : 0;
        
        if (temp_go.min() == temp_go.max()) continue;
        
        size_t Left_n = 0;
        for (size_t j = 0; j < true_cat; j++)
          if (temp_go(j) == 0) Left_n += cat_reduced[j].count;
        if (Left_n < nmin || (N - Left_n) < nmin) continue;
        
        Left_WFail.zeros();
        Left_WRisk.zeros();
        Left_W2Risk.zeros();
        for (size_t j = 0; j < true_cat; j++)
        {
          if (temp_go(j) == 1)
          {
            Left_WFail += cat_reduced[j].WFailCount;
            Left_WRisk += cat_reduced[j].WRiskCount;
            Left_W2Risk += cat_reduced[j].W2RiskCount;
          }
        }
        
        tempscore = suplogrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                                  All_WFail, All_WRisk, All_W2Risk,
                                  All_Fail, All_Risk);
        
        if (tempscore > best_score)
        {
          best_go = temp_go;
          best_score = tempscore;
        }
      }
      
      if (best_score > 0)
      {
        uvec goright(ncat + 1, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          goright(cat_reduced[j].cat) = best_go(j);
        best_cat = pack(ncat + 1, goright);
      }
    }else{
      suplogrank_w_best_cat(cat_reduced, true_cat, ncat, nmin, N,
                            All_Fail, All_Risk,
                            All_WFail, All_WRisk, All_W2Risk,
                            best_cat, best_score, rngl);
    }
  }
  
  if (best_score > TempSplit.score)
  {
    TempSplit.value = best_cat;
    TempSplit.score = best_score;
  }
}


///////////////////////////////////////////////////
// Weighted logrank split functions               //
///////////////////////////////////////////////////

// Weighted logrank score given pre-processed vectors
// U = sum_j [ Left_WFail(j) - pj * All_WFail(j) ]
// V = sum_j [ dj*(nj-dj) / (nj*(nj-1)) * Aj ]
// where pj = Left_WRisk(j) / All_WRisk(j)
//       Aj = (1-pj)^2 * Left_W2Risk(j) + pj^2 * (All_W2Risk(j) - Left_W2Risk(j))
//       dj = All_Fail(j), nj = All_Risk(j) (unweighted counts)
double logrank_w(const vec& Left_WFail,
                 const vec& Left_WRisk,
                 const vec& Left_W2Risk,
                 const vec& All_WFail,
                 const vec& All_WRisk,
                 const vec& All_W2Risk,
                 const uvec& All_Fail,
                 const uvec& All_Risk)
{
  size_t NFail = All_Risk.n_elem - 1;

  double U = 0, V = 0;

  // Loop j = 1 .. NFail inclusive (matching unweighted logrank convention)
  for (size_t j = 1; j <= NFail; j++)
  {
    double nj = All_Risk(j);
    if (nj <= 1) continue;
    if (All_WRisk(j) <= 0) continue;

    double dj = All_Fail(j);
    double pj = Left_WRisk(j) / All_WRisk(j);

    // Score contribution
    U += Left_WFail(j) - pj * All_WFail(j);

    // Variance contribution
    double Aj = (1 - pj) * (1 - pj) * Left_W2Risk(j)
              + pj * pj * (All_W2Risk(j) - Left_W2Risk(j));

    V += dj * (nj - dj) / (nj * (nj - 1)) * Aj;
  }

  if (V > 0)
    return U * U / V;
  else
    return -1;
}

// Weighted suplogrank score: same increments as logrank_w but tracks max|U_cum|
// Q_sup = max|U_cum(j)|^2 / V_final
double suplogrank_w(const vec& Left_WFail,
                    const vec& Left_WRisk,
                    const vec& Left_W2Risk,
                    const vec& All_WFail,
                    const vec& All_WRisk,
                    const vec& All_W2Risk,
                    const uvec& All_Fail,
                    const vec& All_Risk)
{
  size_t NFail = All_Risk.n_elem - 1;

  double U_cum = 0, V_cum = 0, max_abs_U = 0;

  for (size_t j = 1; j <= NFail; j++)
  {
    double nj = All_Risk(j);
    if (nj <= 1) continue;
    if (All_WRisk(j) <= 0) continue;

    double dj = All_Fail(j);
    double pj = Left_WRisk(j) / All_WRisk(j);

    // Score increment (same as logrank)
    double U_inc = Left_WFail(j) - pj * All_WFail(j);
    U_cum += U_inc;

    // Variance increment (same as logrank)
    double Aj = (1 - pj) * (1 - pj) * Left_W2Risk(j)
              + pj * pj * (All_W2Risk(j) - Left_W2Risk(j));
    V_cum += dj * (nj - dj) / (nj * (nj - 1)) * Aj;

    // Track max absolute cumulative score
    double abs_U = U_cum > 0 ? U_cum : -U_cum;
    if (abs_U > max_abs_U)
      max_abs_U = abs_U;
  }

  if (V_cum > 0)
    return max_abs_U * max_abs_U / V_cum;
  else
    return -1;
}

// Weighted logrank score at a numerical cut of x value
double logrank_w_at_x_cut(const uvec& obs_id,
                           const vec& x,
                           const uvec& Y,
                           const uvec& Censor,
                           const size_t NFail,
                           const uvec& All_Fail,
                           const uvec& All_Risk,
                           const vec& All_WFail,
                           const vec& All_WRisk,
                           const vec& All_W2Risk,
                           const vec& obs_weight,
                           double a_random_cut)
{
  size_t N = obs_id.n_elem;
  vec Left_WRisk_raw(NFail + 1, fill::zeros);
  vec Left_WFail(NFail + 1, fill::zeros);
  vec Left_W2Risk_raw(NFail + 1, fill::zeros);

  for (size_t i = 0; i < N; i++)
  {
    if (x(obs_id(i)) <= a_random_cut)
    {
      double wi = obs_weight(i);
      double w2i = wi * wi;
      Left_WRisk_raw(Y(i)) += wi;
      Left_WFail(Y(i)) += wi * Censor(i);
      Left_W2Risk_raw(Y(i)) += w2i;
    }
  }

  // Cumulative from right
  vec Left_WRisk(NFail + 1, fill::zeros);
  vec Left_W2Risk(NFail + 1, fill::zeros);
  Left_WRisk(NFail) = Left_WRisk_raw(NFail);
  Left_W2Risk(NFail) = Left_W2Risk_raw(NFail);
  for (size_t j = NFail; j > 0; j--)
  {
    Left_WRisk(j - 1) = Left_WRisk_raw(j - 1) + Left_WRisk(j);
    Left_W2Risk(j - 1) = Left_W2Risk_raw(j - 1) + Left_W2Risk(j);
  }

  return logrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                   All_WFail, All_WRisk, All_W2Risk,
                   All_Fail, All_Risk);
}

// Weighted logrank score at a random index number, provided with sorted index
double logrank_w_at_id_index(const uvec& indices,
                              const uvec& Y,
                              const uvec& Censor,
                              const size_t NFail,
                              const uvec& All_Fail,
                              const uvec& All_Risk,
                              const vec& All_WFail,
                              const vec& All_WRisk,
                              const vec& All_W2Risk,
                              const vec& obs_weight,
                              size_t a_random_ind)
{
  vec Left_WRisk_raw(NFail + 1, fill::zeros);
  vec Left_WFail(NFail + 1, fill::zeros);
  vec Left_W2Risk_raw(NFail + 1, fill::zeros);

  for (size_t i = 0; i <= a_random_ind; i++)
  {
    double wi = obs_weight(indices(i));
    double w2i = wi * wi;
    Left_WRisk_raw(Y(indices(i))) += wi;
    Left_WFail(Y(indices(i))) += wi * Censor(indices(i));
    Left_W2Risk_raw(Y(indices(i))) += w2i;
  }

  // Cumulative from right
  vec Left_WRisk(NFail + 1, fill::zeros);
  vec Left_W2Risk(NFail + 1, fill::zeros);
  Left_WRisk(NFail) = Left_WRisk_raw(NFail);
  Left_W2Risk(NFail) = Left_W2Risk_raw(NFail);
  for (size_t j = NFail; j > 0; j--)
  {
    Left_WRisk(j - 1) = Left_WRisk_raw(j - 1) + Left_WRisk(j);
    Left_W2Risk(j - 1) = Left_W2Risk_raw(j - 1) + Left_W2Risk(j);
  }

  return logrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                   All_WFail, All_WRisk, All_W2Risk,
                   All_Fail, All_Risk);
}

// Weighted logrank best score (scan all non-tie positions)
void logrank_w_best(const uvec& indices,
                    const uvec& obs_id_sorted,
                    const vec& x,
                    const uvec& Y,
                    const uvec& Censor,
                    const size_t NFail,
                    const uvec& All_Fail,
                    const uvec& All_Risk,
                    const vec& All_WFail,
                    const vec& All_WRisk,
                    const vec& All_W2Risk,
                    const vec& obs_weight,
                    size_t lowindex,
                    size_t highindex,
                    double& temp_cut,
                    double& temp_score)
{
  double score = -1;

  // Maintain raw (non-cumulative) left accumulators
  vec Left_WRisk_raw(NFail + 1, fill::zeros);
  vec Left_WFail(NFail + 1, fill::zeros);
  vec Left_W2Risk_raw(NFail + 1, fill::zeros);

  // Initialize: add all obs up to lowindex
  for (size_t i = 0; i < lowindex; i++)
  {
    double wi = obs_weight(indices(i));
    double w2i = wi * wi;
    Left_WRisk_raw(Y(indices(i))) += wi;
    Left_WFail(Y(indices(i))) += wi * Censor(indices(i));
    Left_W2Risk_raw(Y(indices(i))) += w2i;
  }

  for (size_t i = lowindex; i <= highindex; i++)
  {
    // Move observation into left node
    {
      double wi = obs_weight(indices(i));
      double w2i = wi * wi;
      Left_WRisk_raw(Y(indices(i))) += wi;
      Left_WFail(Y(indices(i))) += wi * Censor(indices(i));
      Left_W2Risk_raw(Y(indices(i))) += w2i;
    }

    // Only evaluate at non-tie boundaries
    if (x(obs_id_sorted(i)) < x(obs_id_sorted(i + 1)))
    {
      // Compute cumulative from right for this candidate
      vec Left_WRisk(NFail + 1, fill::zeros);
      vec Left_W2Risk(NFail + 1, fill::zeros);
      Left_WRisk(NFail) = Left_WRisk_raw(NFail);
      Left_W2Risk(NFail) = Left_W2Risk_raw(NFail);
      for (size_t j = NFail; j > 0; j--)
      {
        Left_WRisk(j - 1) = Left_WRisk_raw(j - 1) + Left_WRisk(j);
        Left_W2Risk(j - 1) = Left_W2Risk_raw(j - 1) + Left_W2Risk(j);
      }

      // Check trivial split
      if (Left_WRisk(0) == 0 || Left_WRisk(0) == All_WRisk(0))
        score = -1;
      else
        score = logrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                          All_WFail, All_WRisk, All_W2Risk,
                          All_Fail, All_Risk);

      if (score > temp_score)
      {
        temp_cut = (x(obs_id_sorted(i)) + x(obs_id_sorted(i + 1))) / 2;
        temp_score = score;
      }
    }
  }
}


///////////////////////////////////////////////////
// Weighted suplogrank scoring functions          //
///////////////////////////////////////////////////

// Weighted suplogrank score at a numerical cut of x value
double suplogrank_w_at_x_cut(const uvec& obs_id,
                              const vec& x,
                              const uvec& Y,
                              const uvec& Censor,
                              const size_t NFail,
                              const uvec& All_Fail,
                              const vec& All_Risk,
                              const vec& All_WFail,
                              const vec& All_WRisk,
                              const vec& All_W2Risk,
                              const vec& obs_weight,
                              double a_random_cut)
{
  size_t N = obs_id.n_elem;
  vec Left_WRisk_raw(NFail + 1, fill::zeros);
  vec Left_WFail(NFail + 1, fill::zeros);
  vec Left_W2Risk_raw(NFail + 1, fill::zeros);

  for (size_t i = 0; i < N; i++)
  {
    if (x(obs_id(i)) <= a_random_cut)
    {
      double wi = obs_weight(i);
      double w2i = wi * wi;
      Left_WRisk_raw(Y(i)) += wi;
      Left_WFail(Y(i)) += wi * Censor(i);
      Left_W2Risk_raw(Y(i)) += w2i;
    }
  }

  vec Left_WRisk(NFail + 1, fill::zeros);
  vec Left_W2Risk(NFail + 1, fill::zeros);
  Left_WRisk(NFail) = Left_WRisk_raw(NFail);
  Left_W2Risk(NFail) = Left_W2Risk_raw(NFail);
  for (size_t j = NFail; j > 0; j--)
  {
    Left_WRisk(j - 1) = Left_WRisk_raw(j - 1) + Left_WRisk(j);
    Left_W2Risk(j - 1) = Left_W2Risk_raw(j - 1) + Left_W2Risk(j);
  }

  return suplogrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                      All_WFail, All_WRisk, All_W2Risk,
                      All_Fail, All_Risk);
}

// Weighted suplogrank score at a random index number
double suplogrank_w_at_id_index(const uvec& indices,
                                 const uvec& Y,
                                 const uvec& Censor,
                                 const size_t NFail,
                                 const uvec& All_Fail,
                                 const vec& All_Risk,
                                 const vec& All_WFail,
                                 const vec& All_WRisk,
                                 const vec& All_W2Risk,
                                 const vec& obs_weight,
                                 size_t a_random_ind)
{
  vec Left_WRisk_raw(NFail + 1, fill::zeros);
  vec Left_WFail(NFail + 1, fill::zeros);
  vec Left_W2Risk_raw(NFail + 1, fill::zeros);

  for (size_t i = 0; i <= a_random_ind; i++)
  {
    double wi = obs_weight(indices(i));
    double w2i = wi * wi;
    Left_WRisk_raw(Y(indices(i))) += wi;
    Left_WFail(Y(indices(i))) += wi * Censor(indices(i));
    Left_W2Risk_raw(Y(indices(i))) += w2i;
  }

  vec Left_WRisk(NFail + 1, fill::zeros);
  vec Left_W2Risk(NFail + 1, fill::zeros);
  Left_WRisk(NFail) = Left_WRisk_raw(NFail);
  Left_W2Risk(NFail) = Left_W2Risk_raw(NFail);
  for (size_t j = NFail; j > 0; j--)
  {
    Left_WRisk(j - 1) = Left_WRisk_raw(j - 1) + Left_WRisk(j);
    Left_W2Risk(j - 1) = Left_W2Risk_raw(j - 1) + Left_W2Risk(j);
  }

  return suplogrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                      All_WFail, All_WRisk, All_W2Risk,
                      All_Fail, All_Risk);
}

// Weighted suplogrank best score (scan all non-tie positions)
void suplogrank_w_best(const uvec& indices,
                       const uvec& obs_id_sorted,
                       const vec& x,
                       const uvec& Y,
                       const uvec& Censor,
                       const size_t NFail,
                       const uvec& All_Fail,
                       const vec& All_Risk,
                       const vec& All_WFail,
                       const vec& All_WRisk,
                       const vec& All_W2Risk,
                       const vec& obs_weight,
                       size_t lowindex,
                       size_t highindex,
                       double& temp_cut,
                       double& temp_score)
{
  double score = -1;

  vec Left_WRisk_raw(NFail + 1, fill::zeros);
  vec Left_WFail(NFail + 1, fill::zeros);
  vec Left_W2Risk_raw(NFail + 1, fill::zeros);

  for (size_t i = 0; i < lowindex; i++)
  {
    double wi = obs_weight(indices(i));
    double w2i = wi * wi;
    Left_WRisk_raw(Y(indices(i))) += wi;
    Left_WFail(Y(indices(i))) += wi * Censor(indices(i));
    Left_W2Risk_raw(Y(indices(i))) += w2i;
  }

  for (size_t i = lowindex; i <= highindex; i++)
  {
    {
      double wi = obs_weight(indices(i));
      double w2i = wi * wi;
      Left_WRisk_raw(Y(indices(i))) += wi;
      Left_WFail(Y(indices(i))) += wi * Censor(indices(i));
      Left_W2Risk_raw(Y(indices(i))) += w2i;
    }

    if (x(obs_id_sorted(i)) < x(obs_id_sorted(i + 1)))
    {
      vec Left_WRisk(NFail + 1, fill::zeros);
      vec Left_W2Risk(NFail + 1, fill::zeros);
      Left_WRisk(NFail) = Left_WRisk_raw(NFail);
      Left_W2Risk(NFail) = Left_W2Risk_raw(NFail);
      for (size_t j = NFail; j > 0; j--)
      {
        Left_WRisk(j - 1) = Left_WRisk_raw(j - 1) + Left_WRisk(j);
        Left_W2Risk(j - 1) = Left_W2Risk_raw(j - 1) + Left_W2Risk(j);
      }

      if (Left_WRisk(0) == 0 || Left_WRisk(0) == All_WRisk(0))
        score = -1;
      else
        score = suplogrank_w(Left_WFail, Left_WRisk, Left_W2Risk,
                             All_WFail, All_WRisk, All_W2Risk,
                             All_Fail, All_Risk);

      if (score > temp_score)
      {
        temp_cut = (x(obs_id_sorted(i)) + x(obs_id_sorted(i + 1))) / 2;
        temp_score = score;
      }
    }
  }
}


///////////////////////////////////////////////////
// Weighted suplogrank dispatcher                  //
///////////////////////////////////////////////////

void Surv_Uni_SupLogrank_Cont(Split_Class& TempSplit,
                              const uvec& obs_id,
                              const vec& x,
                              const uvec& Y,
                              const uvec& Censor,
                              const size_t NFail,
                              const uvec& All_Fail,
                              const vec& All_Risk,
                              int nsplit,
                              double alpha,
                              Rand& rngl,
                              const vec& obs_weight,
                              const vec& All_WFail,
                              const vec& All_WRisk,
                              const vec& All_W2Risk)
{
  size_t N = obs_id.n_elem;
  size_t temp_ind;
  double temp_cut;
  double temp_score = -1;

  if (nsplit > 0) // random split
  {
    for (int k = 0; k < nsplit; k++)
    {
      temp_ind = rngl.rand_sizet(0, N-1);
      temp_cut = x(obs_id(temp_ind));

      temp_score = suplogrank_w_at_x_cut(obs_id, x, Y, Censor, NFail,
                                          All_Fail, All_Risk,
                                          All_WFail, All_WRisk, All_W2Risk,
                                          obs_weight, temp_cut);

      if (temp_score > TempSplit.score)
      {
        TempSplit.value = temp_cut;
        TempSplit.score = temp_score;
      }
    }
    return;
  }

  uvec indices = stable_sort_index(x(obs_id));
  uvec obs_id_sorted = obs_id(indices);

  if (x(obs_id_sorted(0)) == x(obs_id_sorted(N-1))) return;

  size_t lowindex = 0;
  size_t highindex = N - 2;

  if (alpha > 0.5) alpha = 0.5;
  if (alpha > 0)
  {
    size_t nmin = (size_t) N * alpha;
    if (nmin < 1) nmin = 1;
    lowindex = nmin - 1;
    highindex = N - nmin - 1;
  }

  if (x(obs_id_sorted(lowindex)) == x(obs_id_sorted(lowindex+1)) ||
      x(obs_id_sorted(highindex)) == x(obs_id_sorted(highindex+1)))
  {
    check_cont_index_sub(lowindex, highindex, x, obs_id_sorted);
    if (lowindex > highindex) return;
  }


  if (nsplit == 0) // best split
  {
    suplogrank_w_best(indices, obs_id_sorted, x, Y, Censor, NFail,
                      All_Fail, All_Risk,
                      All_WFail, All_WRisk, All_W2Risk,
                      obs_weight, lowindex, highindex,
                      TempSplit.value, TempSplit.score);
    return;
  }
}










