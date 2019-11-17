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

void Surv_Uni_Split_Cat(Uni_Split_Class& TempSplit, 
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
                        bool failforce,
                        size_t ncat)
{
    DEBUG_Rcout << "        --- Surv_One_Split_Cat with ncat = " << ncat << std::endl;
    
    // initiate the failure and at-risk counts for each category 
    std::vector<Surv_Cat_Class> cat_reduced(ncat + 1);
    
    for (size_t j = 1; j < cat_reduced.size(); j++)
        cat_reduced[j].initiate(j, NFail);
    
    for (size_t i = 0; i < obs_id.size(); i++)
    {
        size_t temp_cat = (size_t) x(obs_id(i));
        cat_reduced[temp_cat].count++;
        
        if (Censor(i) == 1)
        {
          cat_reduced[temp_cat].FailCount(Y(i))++; 
          cat_reduced[temp_cat].nfail++;
        }
        
        cat_reduced[temp_cat].RiskCount(Y(i))++;
    }
      
    // count how many nonempty categories

    size_t true_cat = 0;
  
    for (size_t j = 0; j < cat_reduced.size(); j++)
        if (cat_reduced[j].count) true_cat++;

    if (true_cat <= 1)  // nothing to split
        return;    
    
    // move nonzero categories to the front
    
    sort(cat_reduced.begin(), cat_reduced.end(), cat_reduced_collapse);
    
    //for (size_t j = 0; j < cat_reduced.size(); j ++ )
    //  cat_reduced[j].print_simple();
    
    // initiate the total failure and at-risk counts
    
    vec All_Risk(NFail+1, fill::zeros);
    vec All_Fail(NFail+1, fill::zeros);
    
    for (size_t i = 0; i<true_cat; i++)
    {
      All_Fail += cat_reduced[i].FailCount;
      All_Risk += cat_reduced[i].RiskCount;
    }
    
    size_t N = obs_id.n_elem;
    size_t last_count = 0;
    
    for (size_t k = 0; k <= NFail; k++)
    {
      N -= last_count;
      last_count = All_Risk(k);
      All_Risk(k) = N;
    }

    //Rcout << " data here " << join_rows(All_Fail, All_Risk) << std::endl;
    
    // if only two categories, then split on first category
    
    if (true_cat == 2)
    {
      if (split_rule == 1)
        TempSplit.score = logrank(cat_reduced[0].FailCount, cat_reduced[0].RiskCount, All_Fail, All_Risk);
      
      if (split_rule == 2)
        TempSplit.score = suplogrank(cat_reduced[0].FailCount, cat_reduced[0].RiskCount, All_Fail, All_Risk);
      
      if (split_rule > 2)
        Rcout << "      --- splitting rule not implemented yet " << std::endl;
      
      TempSplit.value = record_cat_split(0, cat_reduced);

      return;
    }

    // if more than 2 categories, 

    size_t temp_cat = 0;
    double temp_score = -1;

    vec Left_Fail(NFail+1);    
    vec Left_Risk(NFail+1);
    
    if ( split_gen == 1 or split_gen == 2 )
    {
        for ( int k = 0; k < nsplit; k++ )
        {

          // randomly suffle the categories 
          std::random_shuffle(cat_reduced.begin(), cat_reduced.begin() + true_cat);
          
          //Rcout << "      --- after suffling, get /n " << std::endl;
          //for (size_t j = 0; j < cat_reduced.size(); j ++ )
          //  cat_reduced[j].print_simple();
          
          
          // get low and high index since the categories are re-ordered
          size_t lowindex = 0;
          size_t highindex = true_cat - 2;
          
          if ( alpha > 0 or failforce )
          {
              Rcout << " failforce or alpha in categorical x not implemented yet " << std::endl;
          }          
          
          // randomly select a cut off category           
          size_t temp_cat = (size_t) intRand(lowindex, highindex);
          //Rcout << "      --- got random cut at  " << temp_cat << std::endl;
          
          // calculate left node counts
          Left_Fail.zeros();          
          Left_Risk.zeros();

          for (size_t i = 0; i<= temp_cat; i++)
          {
            Left_Fail += cat_reduced[i].FailCount;
            Left_Risk += cat_reduced[i].RiskCount;
          }
          
          if (split_rule == 1)
            temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
          
          if (split_rule == 2)
            temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
          
          if (split_rule > 2)
            Rcout << "      --- splitting rule not implemented yet " << std::endl;

          if (temp_score > TempSplit.score)
          {
            TempSplit.value = record_cat_split(temp_cat, cat_reduced);
            TempSplit.score = temp_score;
          }
        }
        
        return;
    }
    
    if ( split_gen == 3 ) // best split
    {

      // in case there are too many categories, we gonna shuffle it first, and take the first 1024 choices
      std::random_shuffle(cat_reduced.begin(), cat_reduced.begin() + true_cat);
      
      uvec goright_temp(true_cat, fill::zeros); // this records indicator of the current order of cat_reduced
      goright_temp(0) = 1; // set first cat go right 
      
      size_t counter = 0;
      
      //uvec realcat(true_cat, fill::zeros);
      //for (size_t i =0; i < true_cat; i++)
      //  realcat[i] = cat_reduced[i].cat;
      
      if ( alpha > 0 or failforce )
      {
        Rcout << " failforce or alpha in categorical x not implemented yet " << std::endl;
      }
      
      while(goright_temp(true_cat - 1) == 0 and counter < 1024)
      {
        
        // calculate left node counts
        Left_Fail.zeros();          
        Left_Risk.zeros();
        
        for (size_t i = 0; i<true_cat; i++)
        {
          if (goright_temp(i) == 1)
          {
            Left_Fail += cat_reduced[i].FailCount;
            Left_Risk += cat_reduced[i].RiskCount;            
          }
        }

        
        if (split_rule == 1)
          temp_score = logrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
        
        if (split_rule == 2)
          temp_score = suplogrank(Left_Fail, Left_Risk, All_Fail, All_Risk);
        
        if (split_rule > 2)
          Rcout << "      --- splitting rule not implemented yet " << std::endl;
        
        // Rcout << " Score " << temp_score << " with split \n" << join_rows(realcat, goright_temp) << std::endl;
        
        if (temp_score > TempSplit.score)
        {
          TempSplit.value = record_cat_split(goright_temp, cat_reduced);
          TempSplit.score = temp_score;
        }
        
        // update the splitting rule
        goright_temp(0)++;
        goright_roller(goright_temp);
        
        // update counter
        counter ++;
      }
    }
}


