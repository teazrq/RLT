//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

//using namespace Rcpp;
//using namespace arma;
//using namespace std;

void Surv_Uni_Find_A_Split(Uni_Split_Class& OneSplit,
                           const RLT_SURV_DATA& SURV_DATA,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& RLTParam,
                           uvec& obs_id,
                           uvec& var_id)
{
  DEBUG_Rcout << "    --- Surv_Uni_Find_A_Split " << std::endl;
  
  size_t mtry = Param.mtry;
  size_t nmin = Param.nmin;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  bool usevarweight = Param.usevarweight;
  int varweighttype = Param.varweighttype;
  int nsplit = Param.nsplit;
  int split_gen = Param.split_gen;
  int split_rule = Param.split_rule;
  bool reinforcement = Param.reinforcement;

  size_t N = obs_id.n_elem;
  size_t P = var_id.n_elem;
  //Define PLS parameters
  vec w_etaF;
  vec w_etaC;
  vec z_etaF;
  vec z_etaC;
  vec w_eta(N);
  vec z_eta(N);
  w_eta.ones();
  z_eta.zeros();
  
  // sort obs_id based on Y values 
  const uvec& Y = SURV_DATA.Y;
  const uvec& Censor = SURV_DATA.Censor;
  
  std::sort(obs_id.begin(), obs_id.end(), [Y, Censor](size_t i, size_t j)
  {
      if (Y(i) == Y(j))
          return(Censor(i) > Censor(j));
      else
          return Y(i) < Y(j);
  });
  
  // collapse Y into contiguous integers 
  size_t NFail;
  uvec Y_collapse(N);
  uvec Censor_collapse(N);
  
  DEBUG_Rcout << "    --- Y before collapse \n" << join_rows(Y(obs_id), Censor(obs_id))  << std::endl;
 
  collapse(Y, Censor, Y_collapse, Censor_collapse, obs_id, NFail);
  
  DEBUG_Rcout << "    --- Y after collapse " << std::endl;
  DEBUG_Rcout << join_rows(Y_collapse, Censor_collapse) << std::endl;
  DEBUG_Rcout << "    --- number of failure " << NFail << std::endl;
  
  if (NFail == 0)
    return;   
  
  // initiate the failure and at-risk counts
  uvec All_Risk(NFail+1, fill::zeros);
  uvec All_Fail(NFail+1, fill::zeros);

  for (size_t i = 0; i<N; i++)
  {
    All_Risk(Y_collapse(i)) ++;
      
      if (Censor_collapse(i) == 1)
          All_Fail(Y_collapse(i)) ++;
  }
  
  size_t last_count = 0;
  uvec All_Censor = All_Risk-All_Fail; //The number of times censored observations are repeated
  
  for (size_t k = 0; k <= NFail; k++)
  {
      N -= last_count;
      last_count = All_Risk(k);
      All_Risk(k) = N;
  }

  vec Temp_Vec(NFail+1, fill::zeros);
  
  // if suplogrank, calculate the cc/temp*vterms
  if(split_rule == 2){
    Temp_Vec = 1.0 - conv_to< vec >::from(All_Fail - 1.0)/(All_Risk-1.0); 
    Temp_Vec = Temp_Vec % All_Fail/All_Risk;
    
    for (size_t i =0; i < All_Risk.n_elem; i++)
        if (All_Risk(i) < 2)
            Temp_Vec(i) = 0;
  }

  
  // if logliklihood split, calculate hazard here
  if(split_rule == 3) //or Param.split_rule == 4)
    Temp_Vec = hazard(All_Fail, All_Risk);
  //Rcout << "Hazard: " << Temp_Vec(0)  << std::endl;
  
  uvec obs_id_uni = find_unique(obs_id);
  uvec omega(obs_id_uni.n_elem);
  uvec y_uni(obs_id_uni.n_elem);
  uvec censor_uni(obs_id_uni.n_elem);
  omega.zeros();
  uvec ind;
  
  if(split_rule == 4) //Calculate z & w.  Pass into split function Surv_..._Cont_Pseudo, always be weighted version. Check to make sure pseudo outcomes are calculated correctly.
  {
    // Rcout << obs_id << std::endl;;
    // Rcout << obs_id_uni << std::endl;;
    // for(size_t i = 0; i<Y_collapse.n_elem; i++){
    //   if(any(obs_id_uni==obs_id(i))){
    //     ind = find(obs_id_uni==obs_id(i));
    //     omega(ind(0))++;
    //     y_uni(ind(0)) = Y_collapse(i);
    //     censor_uni(ind(0)) = Censor_collapse(i);
    //   }else{
    //     ind = find(y_uni==Y_collapse(i));
    //     omega(ind(0))++;
    //     // y_uni(ind) = Y_collapse(i);
    //     // censor_uni(ind) = Censor_collapse(i);
    //   }
    // }
    // Rcout << omega << std::endl;;
    // Rcout << y_uni << std::endl;;
    // Rcout << censor_uni << std::endl;;
    // uvec wi(max(obs_id)+1,fill::zeros);
    // 
    // for (size_t k = 0; k < N; k++){
    //   wi(obs_id[k])++;
    // }
    // 
    // vec etaj(N,fill::zeros);
    // 
    // for (size_t i = 0; i<N; i++)
    // {
    //   
    //   All_Risk(Y_collapse(i)) ++;
    //   
    //   if (Censor_collapse(i) == 1)
    //     All_Fail(Y_collapse(i)) ++;
    //   else
    //     
    // }
    vec etaj = conv_to< vec >::from(All_Risk);
    
    //vec tmp = (wi % All_Fail)/etaj;
    //tmp = cumsum(tmp);
    //tmp(All_Risk.n_elem-1) = 0;
    //tmp = shift(tmp, 1);
    
    // vec tmpF = ((All_Fail) % All_Fail)/etaj;
    // tmpF = cumsum(tmpF);
    // tmpF(All_Risk.n_elem-1) = 0;
    // tmpF = shift(tmpF, 1);
    // 
    // vec tmpC = ((All_Censor) % All_Fail)/etaj;
    // tmpC = cumsum(tmpC);
    // tmpC(All_Risk.n_elem-1) = 0;
    // tmpC = shift(tmpC, 1);

    vec tmp = All_Fail/etaj;
    tmp = cumsum(tmp);
    tmp(All_Risk.n_elem-1) = 0;
    tmp = shift(tmp, 1);
    
    //Verion without accounting for ties
    //w_eta = (etaj - 1)/(etaj%etaj);
    
    //Accouting for ties
    // w_etaF = All_Fail % (All_Fail%etaj - All_Fail%All_Fail)/(etaj % etaj);
    // w_etaF = cumsum(w_etaF);
    //Because the set of times where yj is still at risk does not include tj, shift
    //w_etaF(All_Risk.n_elem-1) = 0;
    //w_etaF = shift(w_etaF, 1);
    
    // w_etaC = All_Fail % (All_Censor%etaj - All_Censor%All_Censor)/(etaj % etaj);
    // w_etaC = cumsum(w_etaC);
    //Because the set of times where yj is still at risk includes tj, do not need to shift
    //w_etaC(All_Risk.n_elem-1) = 0;
    //w_etaC = shift(w_etaC, 1);
    
    //vec tmp2F = 1/w_etaF;
    
    // z_etaF = (1/w_etaF) % (All_Fail - tmpF); //Check!!!
    z_etaF = (1 - tmp);
    z_etaF.elem( find_nonfinite(z_etaF) ).zeros();
    // z_etaC = (1/w_etaC) % (0 - tmpF); //Check!!!
    z_etaC = (0 - tmp); //Check!!!
    z_etaC.elem( find_nonfinite(z_etaC) ).zeros();
    //z_etaC(0) = 0;
    
    for (size_t i = 0; i<obs_id.n_elem; i++)
    {
      
      if (Censor_collapse(i) == 1){
        z_eta(i) = z_etaF(Y_collapse(i));
        //w_eta(i) = w_etaF(Y_collapse(i));
      }else{
        z_eta(i) = z_etaC(Y_collapse(i));
        //w_eta(i) = w_etaC(Y_collapse(i));
      }
    }
    
    // Rcout << z_etaF << std::endl;;
    // Rcout << Y_collapse << std::endl;;
    // Rcout << z_eta << std::endl;;

  }
  
  bool failforce = 0; // need to change later 
  double penalty = 0; // initiate 
  
  uvec var_try(P);
  var_try.zeros();
  var_try = var_try + P + 1;
  uword tmp;
  vec var_ws = SURV_DATA.varweight;
  
  // start univariate search
  // shuffle the order of var_id
  // REDO IF SWITCHING ENTIRELY TO ARMA
  if(varweighttype==2){
    //std::default_random_engine generator;
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // std::minstd_rand g1 (seed);  // minstd_rand0 is a standard linear_congruential_engine
    // for(size_t j = 0; j < mtry; j++){
    //   std::discrete_distribution<int> d_mtry{var_ws.begin(),var_ws.end()};
    //   tmp = d_mtry(g1);
    //   var_try(j) = tmp;
    //   var_ws(tmp) = 0;
    // //while(any(var_try==tmp)) tmp = d_mtry(g1);
    // }
    //var_try = Rcpp::RcppArmadillo::sample(var_id, P, false, var_ws);
  }else{
    //var_try = shuffle(var_id);
    var_try = Rcpp::RcppArmadillo::sample(var_id, P, false);
  }
  //Rcout << var_try <<std::endl;;
  
  for (size_t j = 0; j < mtry; j++)
  {
    size_t temp_var = var_try(j);
    
    if (usevarweight) penalty = SURV_DATA.varweight(temp_var); // penalized LL

    Uni_Split_Class TempSplit;
    TempSplit.var = temp_var;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    DEBUG_Rcout << "    --- try var " << temp_var << std::endl;

    if (SURV_DATA.Ncat(temp_var) > 1) // categorical variable 
    {
      DEBUG_Rcout << "      --- try var " << temp_var << " (categorical) " << std::endl;
      
      if (useobsweight)
      {
        Surv_Uni_Split_Cat_W(TempSplit, 
                             obs_id, 
                             SURV_DATA.X.unsafe_col(temp_var), 
                             Y_collapse, 
                             Censor_collapse, 
                             SURV_DATA.obsweight, 
                             NFail,
                             penalty,
                             split_gen, 
                             split_rule, 
                             nsplit, 
                             nmin, 
                             alpha,
                             failforce,
                             SURV_DATA.Ncat(temp_var));

      }else{
        
        Surv_Uni_Split_Cat(TempSplit, 
                           obs_id, 
                           SURV_DATA.X.unsafe_col(temp_var), 
                           Y_collapse, 
                           Censor_collapse,
                           NFail,
                           All_Fail,
                           All_Risk,
                           Temp_Vec,
                           penalty,
                           split_gen, 
                           split_rule, 
                           nsplit, 
                           nmin, 
                           alpha,
                           failforce,
                           SURV_DATA.Ncat(temp_var));
      }

      DEBUG_Rcout << "      --- get var " << temp_var << " at cut " << TempSplit.value << " (categorical) with score " << TempSplit.score << std::endl;
      
      uvec goright(SURV_DATA.Ncat(OneSplit.var) + 1, fill::zeros); 
      unpack(OneSplit.value, SURV_DATA.Ncat(OneSplit.var) + 1, goright);
      
      DEBUG_Rcout << "      --- categories going one side: " << find(goright == 1) << std::endl;
      
      
    }else{ // continuous variable
      //ofstream myfile;
      //myfile.open ("splits.txt", ios::out | ios::app);
      //myfile <<"Var"<<temp_var<< ": ";
      //myfile.close();
      
      DEBUG_Rcout << "      --- try var " << temp_var << " (continuous) " << std::endl;
      
      if (useobsweight)
      {
        Surv_Uni_Split_Cont_W(TempSplit, 
                              obs_id, 
                              SURV_DATA.X.unsafe_col(temp_var), 
                              Y_collapse, 
                              Censor_collapse,
                              SURV_DATA.obsweight, 
                              NFail,
                              penalty,
                              split_gen, 
                              split_rule, 
                              nsplit, 
                              nmin, 
                              alpha,
                              failforce);

      }else{
        
        if(split_rule == 4){
          Surv_Uni_Split_Cont_Pseudo(TempSplit, 
                              obs_id, 
                              SURV_DATA.X.unsafe_col(temp_var), 
                              Y_collapse, 
                              Censor_collapse, 
                              NFail,
                              // w_etaF,//Constant interferes with later calculations
                              // w_etaC,
                              // z_etaF,//Constant interferes with later calculations
                              // z_etaC,//Constant interferes with later calculations
                              w_eta,
                              z_eta,
                              split_gen, 
                              split_rule, 
                              nsplit, 
                              nmin, 
                              alpha,
                              failforce);
        }else{
          Surv_Uni_Split_Cont(TempSplit, 
                              obs_id, 
                              SURV_DATA.X.unsafe_col(temp_var), 
                              Y_collapse, 
                              Censor_collapse, 
                              NFail,
                              All_Fail,
                              All_Risk,
                              Temp_Vec,
                              split_gen, 
                              split_rule, 
                              nsplit, 
                              nmin, 
                              alpha,
                              failforce);
        }
        
      }

      DEBUG_Rcout << "      --- get var " << temp_var << " at cut " << TempSplit.value << " (continuous) with score " << TempSplit.score << std::endl;
    }
    //ofstream myfile;
    //myfile.open ("splits.txt", ios::out | ios::app);
    //myfile <<"\n";
    //myfile.close();
    
    //if(usevarweight) 
    if(varweighttype==1)  TempSplit.score = TempSplit.score*penalty;

    if (TempSplit.score > OneSplit.score)
    {
      OneSplit.var = TempSplit.var;
      OneSplit.value = TempSplit.value;
      OneSplit.score = TempSplit.score;
    }
  }
}







// collapse Y into contiguous integers, Y will always be in an increasing order, failure observations come first

void collapse(const uvec& Y, const uvec& Censor, uvec& Y_collapse, uvec& Censor_collapse, uvec& obs_id, size_t& NFail)
{
    size_t N = obs_id.n_elem;
    
    size_t timepoint = 0;
    size_t current_y = Y[obs_id[0]];
    size_t i = 0;
    
    if (Censor[obs_id[0]] == 1) // no censoring before first failure
    {
        timepoint = 1;
        Y_collapse[0] = 1;
        Censor_collapse[0] = 1;
        i = 1;
    }else{
        while(i < N and Censor[obs_id[i]] == 0)
        {
            Y_collapse[i] = timepoint;
            Censor_collapse[i] = 0;
            current_y = Y[obs_id[i]];
            i++;
            
        }
    }
    
    for(; i < N; i++)
    {
        // a new failure point, update new Y
        if (Y[obs_id[i]] > current_y and Censor[obs_id[i]] == 1)
        {
            timepoint ++;
            current_y = Y[obs_id[i]];
        }
        
        // otherwise, Y_collapse is just the current timepoint
        Y_collapse[i] = timepoint;
        Censor_collapse[i] = Censor[obs_id[i]];
    }
    
    NFail = timepoint;
}
