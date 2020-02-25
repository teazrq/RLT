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
                         const uvec& All_Risk,
                         vec& Temp_Vec,//Constant interferes with later calculations
                         double penalty,
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
    
    // initiate the hazard and log-likelihood for split_rule>2
    // vec lambda0(NFail+1, fill::zeros);
    double Loglik0 = 0;
    // 
    if(split_rule==3 or split_rule==4){
    //   lambda0 = hazard(All_Fail, All_Risk);
       Loglik0 = dot(All_Fail, log(Temp_Vec.replace(0, 1))) - dot(All_Risk, Temp_Vec);
     }
    
    //Rcout << " data here \n" << join_rows(All_Fail, All_Risk) << std::endl;
        
    uvec Left_Risk(NFail+1);
    uvec Left_Fail(NFail+1);
    
    //Rcout << "Starting random split " << std::endl;
    
    if (split_gen == 1) // random split
    {
        for (int k = 0; k < nsplit; k++)
        {
            
            // generate a random cut off
            temp_cut_arma = x(obs_id( (size_t) intRand(0, N-1) ));
            temp_cut = temp_cut_arma(0);
            
            Left_Risk.zeros();
            Left_Fail.zeros();
            
            for (size_t i = 0; i<obs_id.n_elem; i++)
            {
                if (x(obs_id(i)) <= temp_cut)
                {
                    Left_Risk(Y(i)) ++;
                    
                    if (Censor(i) == 1)
                        Left_Fail(Y(i)) ++;
                }
            }
            
            //Rcout << "Calculating Split... "  << std::endl;
            if (split_rule == 1)
                temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
                
            if (split_rule == 2)
                temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec);
            
            if (split_rule == 3 or split_rule == 4)
                temp_score = loglik(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec, Loglik0);
            
            if(split_rule == 4)
                temp_score = temp_score * penalty;
            
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
        
        for (int k = 0; k < nsplit; k++)
        {

            // generate a cut off
            temp_ind = intRand(lowindex, highindex);
            
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
            
            if (split_rule == 3 or split_rule == 4)
              temp_score = loglik(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec, Loglik0);
            
            if(split_rule == 4)
              temp_score = temp_score * penalty;
            
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
            
            if (split_rule == 3 or split_rule == 4)
              temp_score = loglik(Left_Fail, Left_Risk, All_Fail, All_Risk, Temp_Vec, Loglik0);
            
            if(split_rule == 4)
              temp_score = temp_score * penalty;
            
            //Rcout << temp_score << std::endl;
            
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

double logrank(const uvec& Left_Fail, 
               const uvec& Left_Risk, 
               const uvec& All_Fail, 
               const uvec& All_Risk)
{
    uvec Left_Risk_All(Left_Risk.n_elem);
    Left_Risk_All(0) = accu(Left_Risk);
        
    if (Left_Risk_All(0) == 0 or Left_Risk_All(0) == All_Risk(0))
        return -1; 
        
    for (size_t k = 1; k < Left_Risk_All.n_elem; k++)
    {
        Left_Risk_All(k) = Left_Risk_All(k-1) - Left_Risk(k-1);
    }
    
    //Rcout << " \n left node \n" << join_rows(Left_Fail, Left_Risk_All) << std::endl;
    
    vec SizeRatio = conv_to< vec >::from(Left_Risk_All) / All_Risk;
    
    // Variance: N_{1j} / N_{j} * (1 - N_{1j} / N_{j}) * O_{j} * ( N_{j} - O_{j} ) / (N_{j} - 1)
    vec var = SizeRatio % (1.0 - SizeRatio) % All_Fail % (All_Risk - All_Fail) / (All_Risk - 1);
    
    // Difference: O_{1j} - N_{1j} * O_{j} / N_{j}
    vec diff = Left_Fail - SizeRatio % All_Fail;
    
    for (size_t i = 0; i < All_Risk.n_elem; i++)
        if (All_Risk(i) < 2)
            var(i) = 0;
        
    double num = accu(diff);
    
    return num*num/accu(var);
}


double suplogrank(const uvec& Left_Fail, 
                  const uvec& Left_Risk, 
                  const uvec& All_Fail, 
                  const uvec& All_Risk, 
                  vec& Temp_Vec)
{
    uvec Left_Risk_All(Left_Risk.n_elem);
    Left_Risk_All(0) = accu(Left_Risk);
    
    for (size_t k = 1; k < Left_Risk_All.n_elem; k++)
    {
        Left_Risk_All(k) = Left_Risk_All(k-1) - Left_Risk(k-1);
    }
    
    if (Left_Risk_All(0) == 0 or Left_Risk_All(0) == All_Risk(0))
      return -1; 
    
    //Code based on Kosorok Renyi algorithm
    
    // w <- (y1 * y2)/(y1 + y2)
    vec w = (Left_Risk_All % conv_to< vec >::from(All_Risk-Left_Risk_All))/(All_Risk);
    
    // terms <- (d1/y1 - d2/y2)[w > 0]
    vec terms = (conv_to< vec >::from(Left_Fail)/Left_Risk_All-conv_to< vec >::from(All_Fail-Left_Fail)/(All_Risk-Left_Risk_All));   
    
    // check 0 and inf 
    
    for (size_t i = 0; i < All_Risk.n_elem; i++){
      if (All_Risk(i) < 2)
        Temp_Vec(i) = 0;
      if (Left_Risk_All(i) < 1 or (All_Risk(i)-Left_Risk_All(i))<1)
         terms(i) = 0;
    }
      
    double denominator = accu(Temp_Vec % w);
    
    terms = w % terms;
    terms = cumsum(terms);
    
    terms = terms % terms / denominator;

    return max(terms);
}


vec hazard(const uvec& Fail, 
           const uvec& Risk)
{
    //Rcout << "Calculating hazard..."  << std::endl;
    vec haz = conv_to< vec >::from(Fail)/Risk;
    
    //datum::nan replace with 0 handles case where the at-risk
    ///set is empty after a certain timepoint (i.e., all large values of Y are in left node)
    return haz.replace(datum::nan, 0);
}


double loglik(const uvec& Left_Fail, 
              const uvec& Left_Risk, 
              const uvec& All_Fail, 
              const uvec& All_Risk,
              vec& lambda0,
              double& Loglik0)
{
    // cumulative at risk count
    uvec Left_Risk_All(Left_Risk.n_elem);
    Left_Risk_All(0) = accu(Left_Risk);
    
    for (size_t k = 1; k < Left_Risk_All.n_elem; k++)
    {
        Left_Risk_All(k) = Left_Risk_All(k-1) - Left_Risk(k-1);
    }
    
    if (Left_Risk_All(0) == 0 or Left_Risk_All(0) == All_Risk(0))
      return -1; 
    
    //Rcout << "Left_Risk_All(0): " << Left_Risk_All(0)  << std::endl;
    // left and right hazard funcion 
    //vec lambda0 = hazard(All_Fail, All_Risk); 
    vec lambdaLtmp = hazard(Left_Fail, Left_Risk_All);
    vec lambdaRtmp = hazard(All_Fail-Left_Fail, All_Risk-Left_Risk_All);
    
    double epsilon = 0.1;
    
    vec lambdaL = (lambdaLtmp-lambda0)*epsilon+lambda0;//
    vec lambdaR = (lambdaRtmp-lambda0)*epsilon+lambda0;

    //Rcout << "lambdaL(0): " << lambdaL(0)  << std::endl;
    // left and right log-likelihood
    double loglikL = dot(Left_Fail, log(lambdaL.replace(0, 1))) - dot(Left_Risk_All, lambdaL);//Alternatively, dot(Left_Fail, log(lambdaL.replace(0, 1))) - dot(Left_Risk, cumsum(lambdaL))
    double loglikR = dot(All_Fail-Left_Fail, log(lambdaR.replace(0, 1))) - dot(All_Risk-Left_Risk_All, lambdaR);
    double loglik_ep = loglikL + loglikR;
    //double loglik0 = dot(All_Fail, log(lambda0.replace(0, 1))) - dot(All_Risk, lambda0);

    return (loglik_ep-Loglik0)/epsilon;
}