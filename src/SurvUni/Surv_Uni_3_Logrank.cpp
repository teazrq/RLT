//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split on a particular variable
void Surv_Uni_Logrank_Random_Cont(Split_Class& TempSplit,
                                  const uvec& obs_id,
                                  const vec& x,
                                  const uvec& Y, // Y is collapsed
                                  const uvec& Censor, // Censor is collapsed
                                  const size_t NFail,
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
      temp_cut = x(obs_id(temp_ind));
      
      // calculate logrank test score
      temp_score = logrank_at_x_cut(obs_id, x, Y, Censor, NFail, temp_cut);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = temp_cut;
        TempSplit.score = temp_score;
      }
    }
    
    return;
  }
}


//Calculate logrank score at x value cut, sequential calculation without vector
double logrank_at_x_cut(const uvec& obs_id,
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