//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split on a particular variable
void Surv_Uni_Split_Cont(Split_Class& TempSplit,
                        const uvec& obs_id,
                        const vec& x,
                        const uvec& Y, // Y is collapsed
                        const uvec& Censor, // Censor is collapsed
                        const size_t NFail,
                        uvec& All_Fail,
                        vec& All_Risk,
                        vec& Temp_Vec,
                        const vec& obs_weight,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl)
{
  size_t N = obs_id.n_elem;

  arma::vec temp_cut_arma;
  double temp_cut;
  size_t temp_ind;
  double temp_score;
  
  if (split_gen == 1) // random split
  {
    for (int k = 0; k < nsplit; k++)
    {
      // generate a random cut off
      temp_cut_arma = x(obs_id( rngl.rand_sizet(0,N-1) )); 
      temp_cut = temp_cut_arma(0);
      
      // calculate score
      if (useobsweight)
        Rcout << "Weighting not implemented" << std::endl;
      else
        temp_score = surv_cont_score_at_cut(obs_id, x, Y, Censor, NFail, 
                                            All_Fail, All_Risk, Temp_Vec, temp_cut,
                                            split_rule);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = temp_cut;
        TempSplit.score = temp_score;
      }
    }
    
    return;
  }
  
  uvec obs_ranked = sort_index(x(obs_id)); // this is the sorted obs_id by x
  uvec indices = obs_id(sort_index(x(obs_id))); // this is the sorted obs_id by x  
  
  // check identical 
  if ( x(indices(0)) == x(indices(N-1)) ) return;  
  
  // set low and high index
  size_t lowindex = 0; // less equal goes to left
  size_t highindex = N - 2;
  
  // alpha is only effective when x can be sorted
  // I need to do some changes to this
  // this will force nmin for each child node
  if (alpha > 0)
  {
    // if (N*alpha > nmin) nmin = (size_t) N*alpha;
    size_t nmin = (size_t) N*alpha;
    if (nmin < 1) nmin = 1;
    
    lowindex = nmin-1; // less equal goes to left
    highindex = N - nmin - 1;
  }
  
  // if ties
  // move index to better locations
  if ( x(indices(lowindex)) == x(indices(lowindex+1)) or x(indices(highindex)) == x(indices(highindex+1)) )
  {
    check_cont_index_sub(lowindex, highindex, x, indices);
    
    if (lowindex > highindex)
    {
      Rcout << "lowindex > highindex... this shouldn't happen." << std::endl;
      return;
    }
  }

  if (split_gen == 2) // rank split
  {
    for (int k = 0; k < nsplit; k++)
    {
      // generate a cut off
      temp_ind = rngl.rand_sizet( lowindex, highindex); //intRand(lowindex, highindex);
      
      if (useobsweight)
        Rcout << "Weighting not implemented" << std::endl;
      else
        temp_score = surv_cont_score_at_index(indices, obs_ranked, Y, Censor, NFail,
                                              All_Fail, All_Risk,  Temp_Vec, temp_ind,
                                              split_rule);
      
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
    // get score
    if (useobsweight)
      Rcout << "Weighting not implemented" << std::endl;
    else
      surv_cont_score_best(indices, obs_ranked, x, Y, Censor,  NFail,
                           All_Fail, All_Risk, Temp_Vec, lowindex, highindex, 
                           TempSplit.value, TempSplit.score,
                           split_rule);
    
    return;
  }
  
}

//Calculate a score at a random cut
double surv_cont_score_at_cut(const uvec& obs_id,
                        const vec& x,
                        const uvec& Y,
                        const uvec& Censor,
                        const size_t NFail,
                        uvec& All_Fail,
                        vec& All_Risk,
                        vec& Temp_Vec,
                        double a_random_cut,
                        int split_rule)
{
  size_t N = obs_id.size();
  double temp_score;
  
  uvec Left_Risk(NFail+1);
  uvec Left_Fail(NFail+1);
  
  
  Left_Risk.zeros();
  Left_Fail.zeros();
  
  for (size_t i = 0; i<N; i++)
  {
    //If x is less than the random cut, go left
    if (x(obs_id(i)) <= a_random_cut)
    {
      Left_Risk(Y(i)) ++;
      
      if (Censor(i) == 1)
        Left_Fail(Y(i)) ++;
    }
  }
  
  
  // calculate score
  if (split_rule == 1){
    temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
  }
  if (split_rule == 2){
    temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
  }
  
  return temp_score;
  
}


//For rank split
double surv_cont_score_at_index(uvec& indices,
                                uvec& obs_ranked,
                               const uvec& Y,
                               const uvec& Censor,
                               const size_t NFail,
                               uvec& All_Fail,
                               vec& All_Risk,
                               vec& Temp_Vec,
                               size_t a_random_ind,
                               int split_rule)
{
  double temp_score;
  uvec Left_Risk(NFail+1);
  uvec Left_Fail(NFail+1);
  
  Left_Risk.zeros();
  Left_Fail.zeros();
  
  for (size_t i = 0; i <= a_random_ind; i++)
  {
    Left_Risk(Y(obs_ranked(i))) ++;
    
    if (Censor(obs_ranked(i)) == 1)
      Left_Fail(Y(obs_ranked(i))) ++;
  }
  
  if (split_rule == 1)
    temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
  if (split_rule == 2){
    temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
  }
  
  return temp_score;
}


//For best split
void surv_cont_score_best(uvec& indices,
                          uvec& obs_ranked,
                          const vec& x,
                    const uvec& Y,
                    const uvec& Censor,
                    const size_t NFail,
                    uvec& All_Fail,
                    vec& All_Risk,
                    vec& Temp_Vec,
                    size_t lowindex, 
                    size_t highindex, 
                    double& temp_cut, 
                    double& temp_score,
                    int split_rule)
{

  double score;
  uvec Left_Risk(NFail+1);
  uvec Left_Fail(NFail+1);
  
  Left_Risk.zeros();
  Left_Fail.zeros();

  // initiate the failure and censoring counts
  for (size_t i = 0; i<= lowindex; i++)
  {
    Left_Risk(Y(obs_ranked(i))) ++;
    
    if (Censor(obs_ranked(i)) == 1)
      Left_Fail(Y(obs_ranked(i))) ++;
  }
  
  for (size_t i = lowindex; i <= highindex; i++)
  {
    // to use this, highindex cannot be a tie location. 
    // This should be checked already at check_cont_index
    
    while (x(indices(i)) == x(indices(i+1))){
      i++;
      
      Left_Risk(Y(obs_ranked(i))) ++;
      
      if (Censor(obs_ranked(i)) == 1)
        Left_Fail(Y(obs_ranked(i))) ++;
    }
    
    if (split_rule == 1)
      score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
    if (split_rule == 2){
      score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
    }
    
    //If the score has improved, find cut and set new score
    if (score > temp_score)
    {
      temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
      temp_score = score;
    }
    
    if (i + 1 <= highindex)
    {
      Left_Risk(Y(obs_ranked(i+1))) ++;
      
      if (Censor(obs_ranked(i+1)) == 1)
        Left_Fail(Y(obs_ranked(i+1))) ++;
    }

    }

}

double logrank(const uvec& Left_Fail, 
               const uvec& Left_Risk, 
               uvec& All_Fail,
               vec& All_Risk)
{
  uvec Left_Risk_All(Left_Risk.n_elem);
  Left_Risk_All.zeros();
  Left_Risk_All(0) = accu(Left_Risk);
  
  if (Left_Risk_All(0) == 0 or Left_Risk_All(0) == All_Risk(0))
    return -1;
  
  for (size_t k = 1; k < Left_Risk_All.n_elem; k++)
  {
    Left_Risk_All(k) = Left_Risk_All(k-1) - Left_Risk(k-1);
  }
  
  double var = 0;
  double diff = 0;

  for(size_t i=0; i < Left_Risk_All.n_elem; i++){
    if(All_Risk(i)>=2){
      var += Left_Risk_All(i)/All_Risk(i) * (1.0-Left_Risk_All(i)/All_Risk(i)) * All_Fail(i) * 
        (All_Risk(i) - All_Fail(i))/(All_Risk(i)-1);
      diff += Left_Fail(i)- Left_Risk_All(i)/All_Risk(i) * All_Fail(i);
    }
  }

  // Variance: N_{1j} / N_{j} * (1 - N_{1j} / N_{j}) * O_{j} * ( N_{j} - O_{j} ) / (N_{j} - 1)

  // Difference: O_{1j} - N_{1j} * O_{j} / N_{j}
  
  double num = diff*diff;

  return num/var;
}

double suplogrank(const uvec& Left_Fail, 
                  const uvec& Left_Risk, 
                  const uvec& All_Fail, 
                  const vec& All_Risk, 
                  vec& Temp_Vec)
{
  vec Left_Risk_All(Left_Risk.n_elem);
  vec Right_Risk_All(Left_Risk.n_elem);
  vec w(Left_Risk.n_elem);
  vec terms(Left_Risk.n_elem);
  
  Left_Risk_All(0) = accu(Left_Risk);
  Right_Risk_All(0) = All_Risk(0) - Left_Risk_All(0);

  if (Left_Risk_All(0) == 0 or Left_Risk_All(0) == All_Risk(0))
    return -1;     
  
  for (size_t k = 1; k < Left_Risk_All.n_elem; k++)
  {
    Left_Risk_All(k) = Left_Risk_All(k-1) - Left_Risk(k-1);
    Right_Risk_All(k) = All_Risk(k) - Left_Risk_All(k);
    }
  
  //Code based on Kosorok Renyi algorithm

  // w <- (y1 * y2)/(y1 + y2)
  w = (Left_Risk_All % Right_Risk_All)/(All_Risk);
  
  // terms <- (d1/y1 - d2/y2)[w > 0]
  terms = Left_Fail/Left_Risk_All - (All_Fail - Left_Fail)/Right_Risk_All;
  
  // check 0 and inf 

  for (size_t i = 0; i < All_Risk.n_elem; i++){
    if (Left_Risk_All(i) < 1 or Right_Risk_All(i) < 1)
      terms(i) = 0;
  }
  
  double denominator = accu(Temp_Vec % w);
  
  terms = w % terms;
  w = cumsum(terms);
  
  terms = w % w / denominator;

  return max(terms);
}

//Function for the CoxGrad split, since it uses a psuedo outcome
//Simpler than main function
void Surv_Uni_Split_Cont_Pseudo(Split_Class& TempSplit,
                                const uvec& obs_id,
                                const vec& x,
                                const uvec& Y, // Y is collapsed
                                const uvec& Censor, // Censor is collapsed
                                const size_t NFail,
                                vec& z_eta,
                                const vec& obs_weight,
                                double penalty,
                                int split_gen,
                                int nsplit,
                                double alpha,
                                bool useobsweight,
                                Rand& rngl)
{
    size_t N = obs_id.n_elem;
    
    arma::vec temp_cut_arma;
    double temp_cut;
    size_t temp_ind;
    double temp_score;
    
    uvec Pseudo_X(N);
    Pseudo_X.zeros();

    if (split_gen == 1) // random split
    {
      for (int k = 0; k < nsplit; k++)
      {
        
        // generate a random cut off
        temp_cut_arma = x(obs_id( rngl.rand_sizet(0,N-1) )); 
        temp_cut = temp_cut_arma(0);
        
        Pseudo_X.zeros();
        
        //Everything in the left node gets X=1
        for (size_t i = 0; i<N; i++)
        {
          if (x(obs_id(i)) <= temp_cut)
          {
            Pseudo_X(i) = 1;
          }
        }       
        
        temp_score = CoxGrad(Pseudo_X, z_eta)*penalty;
        
        if (temp_score > TempSplit.score)
        {
          TempSplit.value = temp_cut;
          TempSplit.score = temp_score;
        }
      }
      return;
    }
    
    uvec obs_ranked = sort_index(x(obs_id)); // this is the sorted obs_id
    uvec indices = obs_id(sort_index(x(obs_id))); // this is the sorted obs_id by x  
    
    // check identical 
    if ( x(indices(0)) == x(indices(N-1)) ) return;  
    
    // set low and high index
    size_t lowindex = 0; // less equal goes to left
    size_t highindex = N - 2;
    
    // alpha is only effective when x can be sorted
    // this will force nmin for each child node
    if (alpha > 0)
    {
      size_t nmin = (size_t) N*alpha;
      if (nmin < 1) nmin = 1;
      
      lowindex = nmin-1; // less equal goes to left
      highindex = N - nmin - 1;
      
    }
    
    // if ties
    // move index to better locations
    if ( x(indices(lowindex)) == x(indices(lowindex+1)) or x(indices(highindex)) == x(indices(highindex+1)) )
    {
      check_cont_index_sub(lowindex, highindex, x, indices);
      
      if (lowindex > highindex)
      {
        Rcout << "lowindex > highindex... this shouldn't happen." << std::endl;
        return;
      }
    }
    
    
    if (split_gen == 2) // rank split
    {
      
      for (int k = 0; k < nsplit; k++)
      {
        
        // generate a cut off
        temp_ind = rngl.rand_sizet( lowindex, highindex); //intRand(lowindex, highindex);
        
        for (size_t i = 0; i <= temp_ind; i++)
        {
          Pseudo_X(obs_ranked(i)) = 1;
        }
        
        temp_score = CoxGrad(Pseudo_X, z_eta)*penalty;
        
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
      
      // initiate the failure and censoring counts
      for (size_t i = 0; i<= lowindex; i++)
      {
        Pseudo_X(obs_ranked(i)) = 1;
      }
      
      for (size_t i = lowindex; i <= highindex; i++)
      {
        // to use this, highindex cannot be a tie location. 
        // This should be checked already at move_cont_index
        
        while (x(indices(i)) == x(indices(i+1))){
          i++;
          
          Pseudo_X(obs_ranked(i)) = 1;
        }
        
        temp_score = CoxGrad(Pseudo_X, z_eta)*penalty;

        if (temp_score > TempSplit.score)
        {
          TempSplit.value = (x(indices(i)) + x(indices(i+1)))/2;
          TempSplit.score = temp_score;
          
        }
        
        if (i + 1 <= highindex)
        {
          Pseudo_X(obs_ranked(i+1)) = 1;
        }
      }
      
      return;
    }
  }


//Cox gradient split rule implementation
double CoxGrad(uvec& Pseudo_X,
               const vec& z_eta)
{
  //Checking for 0 size node
  if (accu(Pseudo_X) == 0 or accu(Pseudo_X)==Pseudo_X.n_elem)
    return -1;
  
  //Calculate score
  double imp = 0;
  for(size_t i = 0; i<Pseudo_X.n_elem; i++){
    imp += z_eta(i)*Pseudo_X(i);
  }
  return(imp*imp);
}