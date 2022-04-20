//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Surv_Uni_Find_A_Split(Split_Class& OneSplit,
                          const RLT_SURV_DATA& SURV_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id, //We re-order obs_id
                          const uvec& var_id,
                          Rand& rngl)
{
  size_t mtry = Param.mtry;
  size_t nmin = Param.nmin;
  size_t N = obs_id.n_elem;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  bool usevarweight = Param.usevarweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;
  size_t split_rule = Param.split_rule;
  //Define CoxGrad parameters
  vec z_etaF;
  vec z_etaC;
  vec z_eta(N);
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
  collapse(Y, Censor, Y_collapse, Censor_collapse, obs_id, NFail);
  
  if (NFail == 0)
    return;   
  
  // initiate the failure and at-risk counts
  vec All_Risk(NFail+1, fill::zeros);
  uvec All_Fail(NFail+1, fill::zeros);
  
  for (size_t i = 0; i<N; i++)
  {
    All_Risk(Y_collapse(i)) ++;
    
    if (Censor_collapse(i) == 1)
      All_Fail(Y_collapse(i)) ++;
  }
  
  size_t last_count = 0;
  vec All_Censor = All_Risk-All_Fail; //The number of times censored observations are repeated
  
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

  uvec obs_id_uni = find_unique(obs_id);
  uvec omega(obs_id_uni.n_elem);
  uvec y_uni(obs_id_uni.n_elem);
  uvec censor_uni(obs_id_uni.n_elem);
  omega.zeros();
  uvec ind;
  
  if(split_rule == 3) //Calculate z & w.  Pass into split function Surv_..._Cont_Pseudo. 
  {
    vec etaj = All_Risk;
    
    vec tmp = All_Fail/etaj;
    tmp = cumsum(tmp);
    tmp(All_Risk.n_elem-1) = 0;
    tmp = shift(tmp, 1);
    
    z_etaF = (1 - tmp);
    z_etaF.elem( find_nonfinite(z_etaF) ).zeros();
    z_etaC = (0 - tmp);
    z_etaC.elem( find_nonfinite(z_etaC) ).zeros();
    
    for (size_t i = 0; i<obs_id.n_elem; i++)
    {
      
      if (Censor_collapse(i) == 1){
        z_eta(i) = z_etaF(Y_collapse(i));
      }else{
        z_eta(i) = z_etaC(Y_collapse(i));
      }
    }
    
  }
  
  
  // Choose the variables to try
  uvec var_try = rngl.sample(var_id, mtry);

  //For each variable in var_try
  for (auto j : var_try)
  {
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = j;
    TempSplit.value = 0;
    TempSplit.score = -1;
      
    if (SURV_DATA.Ncat(j) > 1) // categorical variable 
    {
      
      // Surv_Uni_Split_Cat(TempSplit, 
      //                   obs_id, 
      //                   SURV_DATA.X.unsafe_col(j), 
      //                   SURV_DATA.Ncat(j),
      //                   Y_collapse, 
      //                   Censor_collapse, 
      //                   NFail,
      //                   All_Fail,
      //                   All_Risk,
      //                   SURV_DATA.obsweight, 
      //                   0.0, // penalty
      //                   split_gen, 
      //                   split_rule, 
      //                   nsplit, 
      //                   alpha, 
      //                   useobsweight,
      //                   rngl);
      
    }else{ // continuous variable
      
      if(split_rule==3){
        Surv_Uni_Split_Cont_Pseudo(TempSplit, 
                                   obs_id, 
                                   SURV_DATA.X.unsafe_col(j), 
                                   Y_collapse, 
                                   Censor_collapse, 
                                   NFail,
                                   z_eta,
                                   SURV_DATA.obsweight,
                                   1.0, // penalty, set to 1 until penalties read in
                                   split_gen, 
                                   nsplit, 
                                   alpha,
                                   useobsweight,
                                   rngl);
      }else{
        Surv_Uni_Split_Cont(TempSplit,
                            obs_id,
                            SURV_DATA.X.unsafe_col(j), 
                            Y_collapse, 
                            Censor_collapse, 
                            NFail,
                            All_Fail,
                            All_Risk,
                            Temp_Vec,
                            SURV_DATA.obsweight,
                            0.0, // penalty
                            split_gen,
                            split_rule,
                            nsplit,
                            alpha,
                            useobsweight,
                            rngl);
      }
      

    }
    
    //If this variable is better than the last one tried
    if (TempSplit.score > OneSplit.score)
    {
      //Change to this variable
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
