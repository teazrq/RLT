//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;


// logrank split socres
void Surv_Uni_Logrank_Cont(Split_Class& TempSplit,
                           const uvec& obs_id,
                           const vec& x,
                           const uvec& Y, // Y is collapsed
                           const uvec& Censor, // Censor is collapsed
                           const size_t NFail,
                           const uvec& All_Fail,
                           const uvec& All_Risk,
                           int split_gen,
                           int nsplit,
                           double alpha,
                           Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t temp_ind;  
  double temp_cut;
  double temp_score = -1;
  
  if (split_gen == 1) // random split
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
  uvec indices = sort_index(x(obs_id)); 
  
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
      RLTcout << "Having difficulty with alpha > 0, maybe due to ties?" << std::endl;
      return;
    }
  }
  
  
  if (split_gen == 2) // rank split
  {
    for (int k = 0; k < nsplit; k++)
    {
      // generate a cut off
      temp_ind = rngl.rand_sizet(lowindex, highindex); //intRand(lowindex, highindex);
      
      // get logrank score at a random index
      temp_score = logrank_at_id_index(indices, Y, Censor, NFail,
                                       All_Fail, All_Risk, temp_ind);
      
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

    logrank_best(indices, obs_id_sorted, x, Y, Censor, NFail,
                 All_Fail, All_Risk, lowindex, highindex, 
                 TempSplit.value, TempSplit.score);
    return;
  }
  
  
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
  size_t last_count = 0;
  size_t all_count = accu(Left_Risk);
  
  if (all_count == 0 or all_count == All_Risk(0))
    return -1;
  
  for (size_t j = 0; j <= NFail; j++)
  {
    all_count -= last_count;
    last_count = Left_Risk(j);
    Left_Risk(j) = all_count;
  }
  
  double Oj = 0, Eij = 0;
  double Nj = 0, Nij = 0;
  double Z = 0, V = 0;
  
  for (size_t j = 1; j < NFail; j++)
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
  
  // RLTcout << "z is " << Z << " v is " << V << std::endl;  
  // RLTcout << "score is " << Z*Z / V << std::endl;  

  return Z*Z / V;

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
  size_t N = indices.n_elem;
  uvec Left_Risk(NFail+1, fill::zeros);
  uvec Left_Fail(NFail+1, fill::zeros);
  
  for (size_t i = 0; i <= a_random_ind; i++)
  {
    Left_Risk(Y(indices(i))) ++;
    Left_Fail(Y(indices(i))) += Censor(indices(i));
  }
  
  // cumulative at risk counts for left
  size_t last_count = 0;
  size_t all_count = accu(Left_Risk);
  
  if (all_count == 0 or all_count == All_Risk(0))
    return -1;
  
  for (size_t j = 0; j <= NFail; j++)
  {
    all_count -= last_count;
    last_count = Left_Risk(j);
    Left_Risk(j) = all_count;
  }
  
  double Oj = 0, Eij = 0;
  double Nj = 0, Nij = 0;
  double Z = 0, V = 0;
  
  for (size_t j = 1; j < NFail; j++)
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
  
  // RLTcout << "z is " << Z << " v is " << V << std::endl;  
  // RLTcout << "score is " << Z*Z / V << std::endl;  
  
  return Z*Z / V;  
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
      score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
      
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
               const uvec& Left_Risk,
               const uvec& All_Fail,
               const uvec& All_Risk)
{

  // cumulative at risk counts for left
  size_t NFail = All_Risk.n_elem - 1;

  uvec Left_Risk_Cum = Left_Risk;
  Left_Risk_Cum(0) = accu(Left_Risk_Cum);
  
  if (Left_Risk_Cum(0) == 0 or Left_Risk_Cum(0) == All_Risk(0))
    return -1;  
  
  for (size_t k = 0; k < NFail; k++)
    Left_Risk_Cum(k+1) = Left_Risk_Cum(k) - Left_Risk(k);  
  
  // doesnt work...
  // uvec Left_Risk_Cum(NFail + 1, fill::zeros);
  // Left_Risk_Cum(NFail) = Left_Risk(NFail);
  // 
  // for (size_t j = NFail-1; j >= 0; j--)
  //   Left_Risk_Cum(j) = Left_Risk_Cum(j+1) + Left_Risk(j);

  double Oj = 0, Eij = 0;
  double Nj = 0, Nij = 0;
  double Z = 0, V = 0;
  
  for (size_t j = 1; j < NFail; j++) // failure times start from 1
  {
    Oj = All_Fail(j);
    Nij = Left_Risk_Cum(j);
    Nj = All_Risk(j);
    Eij = Oj * Nij / Nj;
    Z += Left_Fail(j) - Eij;
    V += Eij * (1 - Oj / Nj) * (Nj - Nij) / (Nj - 1);
  }
  
  Oj = All_Fail(NFail);
  
  // last time point
  if (Oj > 1)
  {
    Nij = Left_Risk_Cum(NFail);
    Nj = All_Risk(NFail);
    Eij = Oj * Nij / Nj;
    Z += Left_Fail(NFail) - Eij;
    V += Eij * (1 - Oj / Nj) * (Nj - Nij) / (Nj - 1);
  }

  return Z*Z / V;
}


//Calculate logrank score at x value cut, sequential calculation without vector
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





