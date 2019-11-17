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
    
    // initiate the failure and at-risk counts
    vec All_Risk(NFail+1, fill::zeros);
    vec All_Fail(NFail+1, fill::zeros);
    
    for (size_t i = 0; i<obs_id.n_elem; i++)
    {
        All_Risk(Y(i)) ++;
        
        if (Censor(i) == 1)
            All_Fail(Y(i)) ++;
    }
    
    size_t N = obs_id.n_elem;
    size_t last_count = 0;

    for (size_t k = 0; k <= NFail; k++)
    {
        N -= last_count;
        last_count = All_Risk(k);
        All_Risk(k) = N;
    }
    
    N = obs_id.n_elem;
    
    
    //Rcout << " data here \n" << join_rows(All_Fail, All_Risk) << std::endl;
        
    vec Left_Risk(NFail+1);
    vec Left_Fail(NFail+1);
    
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
            
            if (split_rule == 1)
                temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
                
            if (split_rule == 2)
                temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
            
            if (split_rule > 2)
                Rcout << "      --- splitting rule not implemented yet " << std::endl;
            
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
                temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
            
            if (split_rule > 2)
                Rcout << "      --- splitting rule not implemented yet " << std::endl;
            
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
                temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
            
            if (split_rule > 2)
                Rcout << "      --- splitting rule not implemented yet " << std::endl;
            
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

double logrank(const vec& Left_Fail, 
               const vec& Left_Risk, 
               const vec& All_Fail, 
               const vec& All_Risk)
{
    vec Left_Risk_All(Left_Risk.n_elem);
    Left_Risk_All(0) = accu(Left_Risk);
        
    if (Left_Risk_All(0) == 0 or Left_Risk_All(0) == All_Risk(0))
        return -1; 
        
    for (size_t k = 1; k < Left_Risk_All.n_elem; k++)
    {
        Left_Risk_All(k) = Left_Risk_All(k-1) - Left_Risk(k-1);
    }
    
    //Rcout << " \n left node \n" << join_rows(Left_Fail, Left_Risk_All) << std::endl;
    
    
    // Variance: Y_{j1} / Y_{j} * (1 - Y_{j1} / Y_{j}) * d_{j} * ( Y_{j} - d_{j} ) / (Y_{j} - 1)
    vec var = Left_Risk_All / All_Risk % (1 - Left_Risk_All / All_Risk) % All_Fail % (All_Risk - All_Fail) / (All_Risk - 1);
    
    // Difference: d_{j1} - Y_{j1} * d_{j} / Y_{j} 
    vec diff = Left_Fail - Left_Risk_All % ( All_Fail / All_Risk );
    
    for (size_t i = 0; i < All_Risk.n_elem; i++)
        if (All_Risk(i) < 2)
            var(i) = 0;
        
    double num = accu(diff);
    
    return num*num/accu(var);
}


double suplogrank(const vec& Left_Fail, 
                  const vec& Left_Risk, 
                  const vec& All_Fail, 
                  const vec& All_Risk)
{
    vec Left_Risk_All(Left_Risk.n_elem);
    Left_Risk_All(0) = accu(Left_Risk);
    
    for (size_t k = 1; k < Left_Risk_All.n_elem; k++)
    {
        Left_Risk_All(k) = Left_Risk_All(k-1) - Left_Risk(k-1);
    }
    
    // Variance: Y_{j1} / Y_{j} * (1 - Y_{j1} / Y_{j}) * d_{j} * ( Y_{j} - d_{j} ) / (Y_{j} - 1)
    vec var = Left_Risk_All / All_Risk % (1 - Left_Risk_All / All_Risk) % All_Fail % (All_Risk - All_Fail) / (All_Risk - 1);
    
    // Difference: d_{j1} - Y_{j1} * d_{j} / Y_{j} 
    vec diff = Left_Fail - Left_Risk_All % ( All_Fail / All_Risk );
    
    for (size_t i = 0; i < All_Risk.n_elem; i++)
        if (All_Risk(i) < 2)
            var(i) = 0;
        
    var = cumsum(var);
    diff = cumsum(diff);
    diff = diff*diff;
    
    return max(diff/var);
}


