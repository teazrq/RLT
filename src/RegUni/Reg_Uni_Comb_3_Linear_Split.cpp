//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split using linear combination
void Reg_Uni_Comb_Linear(Comb_Split_Class& OneSplit,
                         const uvec& split_var,
                         const RLT_REG_DATA& REG_DATA,
                         const PARAM_GLOBAL& Param,
                         const uvec& obs_id,
                         Rand& rngl)
{
  // construct some new data 
  bool useobsweight = Param.useobsweight;
  mat newX(REG_DATA.X(obs_id, split_var));
  vec newY(REG_DATA.Y(obs_id));
  vec newW;
  if (useobsweight) newW = REG_DATA.obsweight(obs_id);
  
  // some parameters
  // there are three split_rule types: sir (default), save, pca, lm
  size_t N = obs_id.n_elem;
  size_t P = split_var.n_elem;
  size_t split_rule = Param.split_rule;
  size_t split_gen = Param.split_gen;
  size_t nsplit = Param.nsplit;
  double alpha = Param.alpha;
  
  // check splitting rule 1 sir; 2 save; 3 pca; 4 lm
  if ( (split_rule == 1 or split_rule == 2) and N < 10)
  {
    split_rule = 4; // switch to lm if sample size is too small
    
    //RLTcout << "sample size too small, switch to lm" << std::endl;
  }
  
  // find splitting rule loading vector 
  vec v;  

  if (split_rule == 1) // default sir
    v = sir(newX, newY, newW, useobsweight, sqrt(N)).col(0);
  
  if (split_rule == 2) // save, not implemented yet
    return;
  
  // pca can be done regardless of sample size
  if (split_rule == 3)
    v = xpc(newX, newW, useobsweight).col(0);
  
  if (split_rule == 4) // lm
  {
    if (useobsweight)
    {
      mat XW = newX;
      XW.each_col() %= sqrt(newW);
      v = solve(XW.t() * XW, newX.t() * (newW % newY), solve_opts::allow_ugly);
    }else{
      v = solve(newX.t() * newX, newX.t() * newY, solve_opts::allow_ugly);
    }
  }
  
  // record splitting variable and loading
  OneSplit.var.subvec(0, P-1) = split_var;
  OneSplit.load.subvec(0, P-1) = v;
  
  // search for the best splitting point with the linear combination
  arma::vec U1 = newX * v;
  
  //Initialize objects
  Split_Class TempSplit;
  TempSplit.var = 0;
  TempSplit.value = 0;
  TempSplit.score = -1;
  
  uvec allid = regspace<uvec>(0,  N-1);
  
  Reg_Uni_Split_Cont(TempSplit,
                     allid,
                     U1,
                     newY,
                     newW,
                     0.0, // penalty
                     split_gen,
                     1, // univariate splitting rule (var)
                     nsplit,
                     alpha,
                     useobsweight,
                     rngl);
  
  OneSplit.value = TempSplit.value;
  OneSplit.score = TempSplit.score;
}


