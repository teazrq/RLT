//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split using linear combination
void Reg_Uni_Comb_Split_Cont(Comb_Split_Class& OneSplit,
                             const uvec& split_var,
                             const vec& split_score,
                             const RLT_REG_DATA& REG_DATA,
                             const PARAM_GLOBAL& Param,
                             const uvec& obs_id,
                             Rand& rngl)
{
  
  RLTcout << "---run Reg_Uni_Comb_Split_Cont on variables \n" << split_var << std::endl;
  
  // construct some new data 
  bool useobsweight = Param.useobsweight;
  mat newX(REG_DATA.X(obs_id, split_var));
  vec newY(REG_DATA.Y(obs_id));
  vec newW;
  if (useobsweight) newW = REG_DATA.obsweight(obs_id);
  
  // some parameters
  // there are three split_rule types: sir (default), pca, save
  size_t N = obs_id.n_elem;
  size_t P = split_var.n_elem;
  size_t split_rule = Param.split_rule;
  size_t split_gen = Param.split_gen;
  size_t nsplit = Param.nsplit;

  
  // check splitting rule 1 sir; 2 save; 3 pca; 4 lm
  if ( (split_rule == 1 or split_rule == 2) and N < 10)
    split_rule = 4; // switch to lm if sample size is too small
  
  // find splitting rule loading vector 
  vec v;  

  if (split_rule == 1) // default sir
  {
    RLTcout << "using SIR split \n" << std::endl;
    
    v = sir(newX, newY, newW, useobsweight, sqrt(N)).col(0);
  }
  
  if (split_rule == 2) // save
  {
    RLTcout << "using SAVE split --- not done yet \n" << std::endl;
    return;
  }
  
  // pca can be done regardless of sample size
  if (split_rule == 3)
  {
    RLTcout << "using PCA split \n" << std::endl;
    
    // pca eigenvectors are sorted with descending eigenvalues
    v = first_pc(newX, newW, useobsweight).col(0);
  }
  
  if (split_rule == 4) // lm
  {
    RLTcout << "using LM split \n" << std::endl;
    
    if (useobsweight)
    {
      mat XW = newX;
      XW.each_col() %= sqrt(newW);
      v = solve(XW.t() * XW, newX.t() * (newW % newY), solve_opts::allow_ugly);
    }else{
      v = solve(newX.t() * newX, newX.t() * newY, solve_opts::allow_ugly);
    }
  }  
  
  
  RLTcout << "---Finish runing sir/save/pca/lm" << std::endl;
  RLTcout << "---obtain loadings " << v << std::endl;
  
  // record splitting variable and loading
  OneSplit.var.subvec(0, P-1) = split_var;
  OneSplit.load.subvec(0, P-1) = v;
  
  // search for the best splitting point with the linear combination
  arma::vec U1 = newX * v;
  
  if (split_gen == 1) // random split
  {
    RLTcout << "-- use random split\n" << std::endl;
    
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a random cut off
      double temp_cut = U1(rngl.rand_sizet(0,N-1));
      double temp_score = -1;
      
      // calculate score
      if (useobsweight)
        temp_score = reg_uni_cont_score_cut_full_w(U1, newY, temp_cut, newW);
      else
        temp_score = reg_uni_cont_score_cut_full(U1, newY, temp_cut);
      
      if (temp_score > OneSplit.score)
      {
        OneSplit.value = temp_cut;
        OneSplit.score = temp_score;
      }
    }
    return;
  }

  // sort data 
  uvec indices = sort_index(U1);
  U1 = U1(indices);
  newY = newY(indices);
  if (useobsweight) newW = newW(indices);
  
  if (U1(0) == U1(U1.n_elem-1)) return;
  
  double alpha = Param.alpha;
  
  // set low and high index
  size_t lowindex = 0; // less equal goes to left
  size_t highindex = N - 2;
  
  // alpha is only effective when x can be sorted
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
  if ( U1(lowindex) == U1(lowindex+1) or U1(highindex) == U1(highindex+1) )
  {
    check_cont_index(lowindex, highindex, (const vec&) U1);
    
    if (lowindex > highindex)
    {
      RLTcout << "lowindex > highindex... this shouldn't happen." << std::endl;
      return;
    }
  }
  
  
  if (split_gen == 2) // rank split
  {
    RLTcout << "-- use rank split\n" << std::endl;
    
    for (size_t k = 0; k < nsplit; k++)
    {
      
      if ( U1(lowindex) == U1(lowindex+1) or U1(highindex) == U1(highindex+1) )
        RLTcout << "still something wrong here " << std::endl;
      
      // generate a cut off
      size_t temp_ind = rngl.rand_sizet( lowindex, highindex );
      double temp_score = -1;
      
      // there could be ties here. need to fix this issue. 
      if (U1(temp_ind) == U1(temp_ind+1))
      {
        //Rcout << "ties at ranked split, move index ..." << std::endl;
        
        if (rngl.rand_01() > 0.5)
        { // move up
          while(U1(temp_ind) == U1(temp_ind+1)) temp_ind++;
        }else{ // move down
          while(U1(temp_ind) == U1(temp_ind+1)) temp_ind--;
        }
      }

      if (useobsweight)
        temp_score = reg_uni_cont_score_rank_full_w(newY, temp_ind, newW);
      else
        temp_score = reg_uni_cont_score_rank_full(newY, temp_ind);
      
      if (temp_score > OneSplit.score)
      {
        OneSplit.value = (U1(temp_ind) + U1(temp_ind+1))/2 ;
        OneSplit.score = temp_score;
      }
    }
    
    return;
  }
  
  
  if (split_gen == 3) // best split
  {
    RLTcout << "-- use best split\n" << std::endl;

    // get score
    if (useobsweight)
      reg_uni_cont_score_best_full_w(U1, newY, newW, lowindex, highindex, OneSplit.value, OneSplit.score);
    else
      reg_uni_cont_score_best_full(U1, newY, lowindex, highindex, OneSplit.value, OneSplit.score);
    
    return;
  }

}

double reg_uni_cont_score_cut_full(const vec& x, 
                                   const vec& y, 
                                   double a_random_cut)
{
  size_t N = x.n_elem;
  
  double LeftSum = 0;
  double RightSum = 0;
  size_t LeftCount = 0;
  
  for (size_t i = 0; i < N; i++)
  {
    //If x is less than the random cut, go left
    if ( x(i) <= a_random_cut )
    {
      LeftCount++;
      LeftSum += y(i);
    }else{
      RightSum += y(i);
    }
  }
  
  // if there are some observations in each node
  if (LeftCount > 0 && LeftCount < N)
    return LeftSum*LeftSum/LeftCount + RightSum*RightSum/(N - LeftCount);
  
  return -1;
}

double reg_uni_cont_score_cut_full_w(const vec& x, 
                                     const vec& y,
                                     double a_random_cut,
                                     const vec& w)
{
  size_t N = x.n_elem;
  
  double LeftSum = 0;
  double RightSum = 0;
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i < N; i++)
  {
    double wi = w(i);
    
    if ( x(i) <= a_random_cut )
    {
      Left_w += wi;
      LeftSum += y(i)*wi;
    }else{
      Right_w += wi;
      RightSum += y(i)*wi;
    }
  }
  
  if (Left_w > 0 && Right_w < N)
    return LeftSum*LeftSum/Left_w + RightSum*RightSum/Right_w;
  
  return -1;
  
}

double reg_uni_cont_score_rank_full(const vec& y,
                                    size_t a_random_ind)
{
  size_t N = y.n_elem;
  
  double LeftSum = 0;
  double RightSum = 0;  

  //Count the number of observations with a smaller or equal index
  for (size_t i = 0; i <= a_random_ind; i++)
    LeftSum += y(i);
  
  //Count the other observations
  for (size_t i = a_random_ind+1; i < N; i++)
    RightSum += y(i);
  
  return LeftSum*LeftSum/(a_random_ind + 1) + RightSum*RightSum/(N - a_random_ind - 1);
}

double reg_uni_cont_score_rank_full_w(const vec& y,
                                      size_t a_random_ind,
                                      const vec& w)
{
  size_t N = y.n_elem;
  
  double LeftSum = 0;
  double RightSum = 0;
  double Left_w = 0;
  double Right_w = 0;
  
  //Count the number of observations with a smaller or equal index
  for (size_t i = 0; i <= a_random_ind; i++)
  {
    LeftSum += y(i)*w(i);
    Left_w += w(i);
  }
  
  //Count the other observations
  for (size_t i = a_random_ind+1; i < N; i++)
  {
    RightSum += y(i)*w(i);
    Right_w += w(i);
  }
  
  return LeftSum*LeftSum/Left_w + RightSum*RightSum/Right_w;
}


//For best split
void reg_uni_cont_score_best_full(const vec& x,
                                  const vec& y,
                                  size_t lowindex, 
                                  size_t highindex, 
                                  double& temp_cut, 
                                  double& temp_score)
{
  size_t N = y.n_elem;
  double score = 0;
  
  double LeftSum = 0;
  double RightSum = 0;
  
  //Find left or right of the lowindex to start
  for (size_t i = 0; i <= lowindex; i++)
    LeftSum += y(i);
  
  for (size_t i = lowindex+1; i < N; i++)
    RightSum += y(i);
  
  //Trying the other splits
  for (size_t i = lowindex; i <= highindex; i++)
  {
    
    //If there is a tie
    while (x(i) == x(i+1)){
      i++;
      
      //Adjust sums
      LeftSum += y(i);
      RightSum -= y(i);
    }
    
    //Calculate score
    score = LeftSum*LeftSum/(i + 1) + RightSum*RightSum/(N - i - 1);
    
    //If the score has improved, find cut and set new score
    if (score > temp_score)
    {
      temp_cut = (x(i) + x(i + 1))/2 ;
      temp_score = score;
    }
    
    //Adjust sums
    if (i + 1 <= highindex)
    {
      LeftSum += y(i+1);
      RightSum -= y(i+1);
    }
  }
}



void reg_uni_cont_score_best_full_w(const vec& x,
                                    const vec& y,
                                    const vec& w,
                                    size_t lowindex, 
                                    size_t highindex, 
                                    double& temp_cut, 
                                    double& temp_score)
{
  size_t N = y.n_elem;
  double score = 0;
  
  double LeftSum = 0;
  double RightSum = 0;
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i <= lowindex; i++)
  {
    LeftSum += y(i)*w(i);
    Left_w += w(i);
  }
  
  for (size_t i = lowindex+1; i < N; i++)
  {
    RightSum += y(i)*w(i);
    Right_w += w(i);
  }
  
  for (size_t i = lowindex; i <= highindex; i++)
  {
    while (x(i) == x(i+1)){
      i++;
      
      LeftSum += y(i)*w(i);
      RightSum -= y(i)*w(i);
      
      Left_w += w(i);
      Right_w += w(i);
    }
    
    score = LeftSum*LeftSum/Left_w + RightSum*RightSum/Right_w;
    
    if (score > temp_score)
    {
      temp_cut = (x(i) + x(i + 1))/2 ;
      temp_score = score;
    }
    
    if (i + 1 <= highindex)
    {
      LeftSum += y(i+1)*w(i+1);
      RightSum -= y(i+1)*w(i+1);
      
      Left_w += w(i+1);
      Right_w += w(i+1);
    }
  }
}