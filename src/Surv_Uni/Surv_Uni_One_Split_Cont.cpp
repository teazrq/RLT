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
                         const uvec& Y,
                         const uvec& Censor,
                         double penalty,
                         int split_gen,
                         int split_rule,
                         int nsplit,
                         size_t nmin, 
                         double alpha,
                         vec& obs_weight,
                         bool useobsweight,
                         size_t NFail,
                         int failforce)
{
    size_t N = obs_id.n_elem;
    
    arma::vec temp_cut_arma;
    double temp_cut;
    size_t temp_ind;
    double temp_score;
    
    if (split_gen == 1) // random split
    {
        DEBUG_Rcout << "      --- Surv_One_Split_Cont with " << nsplit << " random split " << std::endl;
        
        uvec Left_Count_Fail(NFail+1);
        uvec Left_Count_Censor(NFail+1);
        uvec Right_Count_Fail(NFail+1);
        uvec Right_Count_Censor(NFail+1);
        size_t LeftN; 
        
        for (int k = 0; k < nsplit; k++)
        {
            // reset the counts 
            Left_Count_Fail.zeros();
            Left_Count_Censor.zeros();
            Right_Count_Fail.zeros();
            Right_Count_Censor.zeros();
            LeftN = 0;             
            
            // generate a random cut off
            temp_cut_arma = x(obs_id( (size_t) intRand(0, N-1) ));
            temp_cut = temp_cut_arma(0);

            // initiate the failure and censoring counts
            for (size_t i = 0; i<N; i++)
            {
                if (x[obs_id[i]] <= temp_cut) // go left
                {
                    if (Censor[i] == 1)
                        Left_Count_Fail[Y[i]]++;
                    else
                        Left_Count_Censor[Y[i]]++;
                    
                    LeftN++;
                }else{  // go right
                    if (Censor[i] == 1)
                        Right_Count_Fail[Y[i]]++;
                    else
                        Right_Count_Censor[Y[i]]++;
                }
            }

            if (useobsweight){
                DEBUG_Rcout << "      --- weighted survival not implemented yet " << std::endl;
            }else{
                
                temp_score = logrank(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, LeftN, N, NFail);
                DEBUG_Rcout << "      --- calculate logrank temp_score " << temp_score << std::endl;
            }
            
            if (temp_score > TempSplit.score)
            {
                TempSplit.value = temp_cut;
                TempSplit.score = temp_score;
            }
        }
        
        DEBUG_Rcout << "      --- Best cut off at " << TempSplit.value << " with score " << TempSplit.score << std::endl;
        return;
    }
}

double logrank(uvec& Left_Count_Fail,
               uvec& Left_Count_Censor,
               uvec& Right_Count_Fail,
               uvec& Right_Count_Censor,
               size_t LeftN_i,
               size_t N_i,
               size_t nfail)
{
    double numerator = 0;
    double denominator = 0;
    double tempscore = -1;
    
    double LeftN = (double) LeftN_i;
    double N = (double) N_i;
    
    // calculate the logrank for this split
    LeftN -= Left_Count_Censor[0];
    N -= Left_Count_Censor[0] + Right_Count_Censor[0];    
    
    for (size_t j = 1; j <= nfail && N > 1; j++)
    {
        numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/N - Left_Count_Fail[j];
        denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/N*(1- LeftN/N)*(N - Left_Count_Fail[j] - Right_Count_Fail[j])/(N - 1);

        LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
        N -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];
    }
    
    if (denominator > 0)
        tempscore = numerator*numerator/denominator;
    
    return tempscore;
}