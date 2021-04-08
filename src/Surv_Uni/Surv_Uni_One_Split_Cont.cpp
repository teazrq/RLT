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

void Surv_Uni_Split_Cont(Uni_Split_Class& TempSplit, 
                         uvec& obs_id,
                         const vec& x,
                         const uvec& Y, // Y is collapsed
                         const uvec& Censor, // Censor is collapsed
                         size_t NFail,
                         const uvec& All_Fail,
                         const vec& All_Risk,
                         vec& Temp_Vec,//Constant interferes with later calculations
                         int split_gen,
                         int split_rule,
                         int nsplit,
                         size_t nmin, 
                         double alpha,
                         bool failforce)
{

    vec temp_cut_arma;
    double temp_cut;
    size_t temp_ind;
    double temp_score = -1;
    size_t N = obs_id.n_elem;

    //Rcout << " data here \n" << join_rows(All_Fail, All_Risk) << std::endl;
        
    uvec Left_Risk(NFail+1);
    uvec Left_Fail(NFail+1);
    
    //Rcout << "Starting random split " << std::endl;
    
    if (split_gen == 1) // random split
    {
        for (int k = 0; k < nsplit; k++)
        {

            // generate a random cut off
            temp_cut_arma = x(obs_id((size_t) intRand(0, N-1) ));
            temp_cut = temp_cut_arma(0);//x(obs_id((size_t) temp_cut_arma2(k)));

            Left_Risk.zeros();
            Left_Fail.zeros();

            for (size_t i = 0; i<N; i++)
            {
                if (x(obs_id(i)) <= temp_cut)
                {
                    Left_Risk(Y(i)) ++;

                    if (Censor(i) == 1)
                        Left_Fail(Y(i)) ++;
                }
            }

            //Rcout << "Calculating Split... "  << std::endl;
            if (split_rule == 1){
              temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
            }

            if (split_rule == 2)
                temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);

            if (temp_score > TempSplit.score)
            {
                TempSplit.value = temp_cut;
                TempSplit.score = temp_score;
            }
            //Rcout << "Finishing Split... "  << std::endl;
        }
        return;
    }
    
    uvec obs_ranked = sort_index(x(obs_id)); // this is the sorted obs_id
    uvec indices = obs_id(obs_ranked); // this is the sorted obs_id
    
    // check identical 
    if ( x(indices(0)) == x(indices(N-1)) ) return;  
    
    // set low and high index
    size_t lowindex = 0; // less equal goes to left
    size_t highindex = N - 2;
    
    // alpha is only effective when x can be sorted
    // this will force nmin for each child node
    if (alpha > 0)
    {
        if (N*alpha > nmin) nmin = (size_t) N*alpha;
        
        // if there are ties, do further check
        if ( (x(indices(lowindex)) == x(indices(lowindex + 1))) | (x(indices(highindex)) == x(indices(highindex + 1))) )
            move_cont_index(lowindex, highindex, x, indices, nmin);
        
    }else{
        // move index if ties
        while( x(indices(lowindex)) == x(indices(lowindex + 1)) ) lowindex++;
        while( x(indices(highindex)) == x(indices(highindex + 1)) ) highindex--;    
        
        if (lowindex > highindex) return;
    }
    
    // force number of failures
    
    if (failforce)
    {
        Rcout << " failforce not implemented yet " << std::endl;
    }
    
    
    if (split_gen == 2) // rank split
    {
        DEBUG_Rcout << "      --- Surv_One_Split_Cont with " << nsplit << " rank split " << std::endl;
        
        Vector<INTSXP> temp_ind_all = sample(highindex-lowindex, nsplit)+lowindex-1;
        
        for (int k = 0; k < nsplit; k++)
        {

            // generate a cut off
            temp_ind = (size_t) temp_ind_all(k);
          
            Left_Risk.zeros();
            Left_Fail.zeros();
            
            for (size_t i = 0; i <= temp_ind; i++)
            {
                Left_Risk(Y(obs_ranked(i))) ++;
                
                if (Censor(obs_ranked(i)) == 1)
                    Left_Fail(Y(obs_ranked(i))) ++;
            }
            
            if (split_rule == 1)
                temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
            
            if (split_rule == 2)
                temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
            
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
            // This should be checked already at move_cont_index
            
            while (x(indices(i)) == x(indices(i+1))){
                i++;
                
                Left_Risk(Y(obs_ranked(i))) ++;
                
                if (Censor(obs_ranked(i)) == 1)
                    Left_Fail(Y(obs_ranked(i))) ++;
            }
            
            if (split_rule == 1)
                temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
            
            if (split_rule == 2)
                temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
            
            if (temp_score > TempSplit.score)
            {
                TempSplit.value = (x(indices(i)) + x(indices(i+1)))/2;
                TempSplit.score = temp_score;
                
            }
            
            if (i + 1 <= highindex)
            {
                Left_Risk(Y(obs_ranked(i+1))) ++;
                
                if (Censor(obs_ranked(i+1)) == 1)
                    Left_Fail(Y(obs_ranked(i+1))) ++;
            }
        }

        return;
    }
}

//Function for the CoxGrad split, since it uses a psuedo outcome
//Simpler than main function
void Surv_Uni_Split_Cont_Pseudo(Uni_Split_Class& TempSplit, 
                         uvec& obs_id,
                         const vec& x,
                         const uvec& Y, // Y is collapsed
                         const uvec& Censor, // Censor is collapsed
                         size_t NFail,
                         vec& z_eta,
                         int split_gen,
                         int nsplit,
                         size_t nmin, 
                         double alpha,
                         bool failforce)
{
  
  vec temp_cut_arma;
  double temp_cut;
  size_t temp_ind;
  double temp_score = -1;
  size_t N = obs_id.n_elem;

  //Rcout << " data here \n" << join_rows(All_Fail, All_Risk) << std::endl;
  
  uvec Pseudo_X(N);
  Pseudo_X.zeros();
  
  //Rcout << "Starting random split " << std::endl;
  
  if (split_gen == 1) // random split
  {
    for (int k = 0; k < nsplit; k++)
    {
       
      temp_cut_arma = x(obs_id((size_t) intRand(0, N-1) ));
      temp_cut = temp_cut_arma(0);
      //Rcout << "Splitting at "<<temp_cut << std::endl;
       
      Pseudo_X.zeros();
       
       //Everything in the left node gets X=1
       for (size_t i = 0; i<N; i++)
       {
         if (x(obs_id(i)) <= temp_cut)
         {
           Pseudo_X(i) = 1;
         }
       }       
    //   //Rcout << "Calculating Split... "  << std::endl;
    //   
       temp_score = CoxGrad(Pseudo_X, z_eta);
     
       if (temp_score > TempSplit.score)
       {
         TempSplit.value = temp_cut;
         TempSplit.score = temp_score;
       }
    }
    return;
  }
  
  uvec obs_ranked = sort_index(x(obs_id)); // this is the sorted obs_id
  uvec indices = obs_id(obs_ranked); // this is the sorted obs_id
  
  // check identical 
  if ( x(indices(0)) == x(indices(N-1)) ) return;  
  
  // set low and high index
  size_t lowindex = 0; // less equal goes to left
  size_t highindex = N - 2;
  
  // alpha is only effective when x can be sorted
  // this will force nmin for each child node
  if (alpha > 0)
  {
    if (N*alpha > nmin) nmin = (size_t) N*alpha;
    
    // if there are ties, do further check
    if ( (x(indices(lowindex)) == x(indices(lowindex + 1))) | (x(indices(highindex)) == x(indices(highindex + 1))) )
      move_cont_index(lowindex, highindex, x, indices, nmin);
    
  }else{
    // move index if ties
    while( x(indices(lowindex)) == x(indices(lowindex + 1)) ) lowindex++;
    while( x(indices(highindex)) == x(indices(highindex + 1)) ) highindex--;    
    
    if (lowindex > highindex) return;
  }
  
  // force number of failures
  
  if (failforce)
  {
    Rcout << " failforce not implemented yet " << std::endl;
  }
  
  
  if (split_gen == 2) // rank split
  {
    DEBUG_Rcout << "      --- Surv_One_Split_Cont with " << nsplit << " rank split " << std::endl;
    
    for (int k = 0; k < nsplit; k++)
    {
      
      // generate a cut off
      temp_ind = (size_t) sample(N, 1)(0)-1;
      
      for (size_t i = 0; i <= temp_ind; i++)
      {
        Pseudo_X(obs_ranked(i)) = 1;
      }
      
        temp_score = CoxGrad(Pseudo_X, z_eta);

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
      
        temp_score = CoxGrad(Pseudo_X, z_eta);

      //Rcout << "Score at "<<(x(indices(i)) + x(indices(i+1)))/2 << ": "<<temp_score << std::endl;
      
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

//Comments include old, slower logrank version
double logrank(const uvec& Left_Fail, 
               const uvec& Left_Risk, 
               const uvec& All_Fail, 
               const vec& All_Risk)
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
    
    // uvec Left_Risk_All = shift(reverse(cumsum(Left_Risk)))
    // Left_Risk_All(0) = accu(Left_Risk);
    
    //Rcout << " \n left node \n" << join_rows(Left_Fail, Left_Risk_All) << std::endl;
    
    double var = 0;
    double diff = 0;

    //vec SizeRatio = conv_to< vec >::from(Left_Risk_All) / All_Risk;
    
    for(size_t i=0; i < Left_Risk_All.n_elem; i++){
      if(All_Risk(i)>=2){
        var += Left_Risk_All(i)/All_Risk(i) * (1.0-Left_Risk_All(i)/All_Risk(i)) * All_Fail(i) * 
          (All_Risk(i) - All_Fail(i))/(All_Risk(i)-1);
      }
      diff += Left_Fail(i)- Left_Risk_All(i)/All_Risk(i) * All_Fail(i);
    }
    //Check Armadillo for difference between [] and ()
    //Can make them the same by turning off boundary check
    
    
    // Variance: N_{1j} / N_{j} * (1 - N_{1j} / N_{j}) * O_{j} * ( N_{j} - O_{j} ) / (N_{j} - 1)
    //vec var = SizeRatio % (1.0 - SizeRatio) % All_Fail % (All_Risk - All_Fail) / (All_Risk - 1);
    
    // Difference: O_{1j} - N_{1j} * O_{j} / N_{j}
    //vec diff = Left_Fail - SizeRatio % All_Fail;
    
    //uvec small = find(All_Risk < 2);
    //A.elem( find(A > 0.5) ).ones();
    //var.elem( small ) = zeros<vec>(small.n_elem);
    //for (size_t i = 0; i < All_Risk.n_elem; i++)
    //    if (All_Risk(i) < 2)
    //        var(i) = 0;
    
        
    double num = diff;//accu(diff);
    
    //Rcout << num*num/var << std::endl;
    
    return num*num/var;//accu(var);
}


double suplogrank(const uvec& Left_Fail, 
                  const uvec& Left_Risk, 
                  const uvec& All_Fail, 
                  const vec& All_Risk, 
                  vec& Temp_Vec)
{
    uvec Left_Risk_All(Left_Risk.n_elem);
    Left_Risk_All(0) = accu(Left_Risk);
    
    if (Left_Risk_All(0) == 0 or Left_Risk_All(0) == All_Risk(0))
      return -1;     
    
    for (size_t k = 1; k < Left_Risk_All.n_elem; k++)
    {
        Left_Risk_All(k) = Left_Risk_All(k-1) - Left_Risk(k-1);
    }
    
    //Code based on Kosorok Renyi algorithm
    
    vec Right_Risk_All = conv_to< vec >::from(All_Risk - Left_Risk_All);
    
    // w <- (y1 * y2)/(y1 + y2)
    vec w = (Left_Risk_All % Right_Risk_All)/(All_Risk);
    
    // terms <- (d1/y1 - d2/y2)[w > 0]
    vec terms = conv_to< vec >::from(Left_Fail)/Left_Risk_All - (All_Fail - Left_Fail)/Right_Risk_All;   
    
    // check 0 and inf 
    
    for (size_t i = 0; i < All_Risk.n_elem; i++){
      if (Left_Risk_All(i) < 1 or Right_Risk_All(i) < 1)
         terms(i) = 0;
    }
      
    double denominator = accu(Temp_Vec % w);
    
    terms = w % terms;
    terms = cumsum(terms);
    
    terms = terms % terms / denominator;

    return max(terms);
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