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
                             const uvec& use_var,
                             const RLT_REG_DATA& REG_DATA, 
                             const PARAM_GLOBAL& Param,
                             const uvec& obs_id,
                             Rand& rngl)
{
  
  RLTcout << "Use comb cont split with variables \n" << use_var << std::endl;
 
  // construct some new data 
  bool useobsweight = Param.useobsweight;
  mat newX(REG_DATA.X(obs_id, use_var));
  vec newY(REG_DATA.Y(obs_id));
  vec newW;
  if (useobsweight) newW = REG_DATA.obsweight(obs_id);
  
  // some parameters
  // there are three split_rule types: sir (default), save, and pca
  size_t N = obs_id.n_elem;
  size_t P = use_var.n_elem;
  size_t split_rule = Param.split_rule;
  size_t split_gen = Param.split_gen;
  size_t nsplit = Param.nsplit;

  // find splitting rule 
  vec v;
  
  if (split_rule == 1 and N >= 15) // default sir
  {
    RLTcout << "using SIR split --- not done yet, switching to pca \n" << std::endl;
    split_rule = 3;
    
    //mat M = sir(newX, newY, newW, useobsweight, sqrt(N));
    
    //mat xcov = wcov(newX, newW, useobsweight);
    
    //vec eigval;
    //mat eigvec;
    //eig_sym(eigval, eigvec, M);
    
    // eigenvalues are ascending
    //v = eigvec.col( eigvec.n_cols );
  }
  
  if (split_rule == 2 and N >= 15) // save
  {
    RLTcout << "using SAVE split --- not done yet, switching to pca \n" << std::endl;
    split_rule = 3;
    
    //mat M = save(newX, newY, newW, useobsweight, sqrt(N));
    
    //vec eigval;
    //mat eigvec;
    //eig_sym(eigval, eigvec, M);
    
    // eigenvalues are ascending
    //v = eigvec.col( eigvec.n_cols );
  }
  
  if (split_rule == 3 or N < 15) // pca
  {
    RLTcout << "using PCA split \n" << std::endl;

    mat coeff = princomp(newX);
    
    // eigenvalues are descending
    v = coeff.col(0);
  }
  
  // record splitting variable and loading
  OneSplit.var.subvec(0, P-1) = use_var;
  OneSplit.load.subvec(0, P-1) = v;
  
  // search for the best split
  arma::vec U1 = newX * v;
  //Rcout << "new linear combination x \n" << U1 << std::endl;
  
  if (split_gen == 1) // random split
  {
    RLTcout << "random splitting \n" << std::endl;
    
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
      
      RLTcout << "Try cut " << temp_cut << " with score " << temp_score << std::endl;
      
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
    RLTcout << "Rank split\n" << std::endl;
    
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
    RLTcout << "Best split\n" << std::endl;
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


arma::mat sir(arma::mat& newX, 
              arma::vec& newY, 
              arma::vec& newW,
              bool useobsweight,
              size_t nslice)
{
  uvec index = sort_index(newY);
  size_t N = newY.n_elem;
  size_t P = newX.n_cols;

  mat M(P, P, fill::zeros);
  
  mat sortedx = newX.rows(index);
  vec sortedy = newY(index);
  
  size_t slice_size = (size_t) N/nslice;
  size_t res = N - nslice*slice_size;
  
  if (useobsweight)
  {
    // apply weights
    sortedx.each_col() %= newW;
    rowvec xbar = sum(sortedx, 0) / sum(newW);
    
    // obs index
    size_t rownum = 0;
    
    for (size_t k =0; k < nslice; k++)
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice weight
      double wh = sum(newW.subvec(rownum, rownum + nh - 1));

      // weighted slice mean
      rowvec xhbar = sum(sortedx.rows(rownum, rownum + nh - 1), 0) / wh;
      
      // next slice
      rownum += nh;

      // add to estimation matrix 
      M += (xhbar - xbar).t() * (xhbar - xbar) * wh;
    }
    
  }else{
    // x mean
    rowvec xbar = mean(sortedx, 0);
      
    // obs index
    size_t rownum = 0;
    
    for (size_t k =0; k < nslice; k++) 
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice mean
      rowvec xhbar = mean(sortedx.rows(rownum, rownum + nh - 1), 0);

      // next slice
      rownum += nh;
      
      // add to estimation matrix 
      M += (xhbar - xbar).t() * (xhbar - xbar) * nh;
    }
  }
  
  return M;
}

// arma::mat wcov(arma::mat& newX, 
//                arma::vec& newW, 
//                bool useobsweight)
// {
//   if (useobsweight)
//   {
//     // apply weights
//     // weighted x mean
//     rowvec xwbar = sum(newX.each_col() % newW, 0) / sum(newW);
//     
//     // cov
//     mat xrw = newX.each_row() - xwbar;
//     mat xrw = newX.each_col() % sqrt(newW);    
//     
//     return xrw.t() * xrw / (1.0 -  norm(newW, "fro"));
// 
//   }else{
//     return var(newX);
//   }
// }



arma::mat save(arma::mat& newX, 
               arma::vec& newY, 
               arma::vec& newW,
               bool useobsweight,
               size_t nslice)
{
  uvec index = sort_index(newY);
  size_t N = newY.n_elem;
  size_t P = newX.n_cols;
  
  mat M(P, P, fill::zeros);
  
  mat sortedx = newX.rows(index);
  vec sortedy = newY(index);
  
  size_t slice_size = (size_t) N/nslice;
  size_t res = N - nslice*slice_size;
  mat Diag(P, P, fill::eye);
  
  if (useobsweight)
  {
    
    // obs index
    size_t rownum = 0;
    
    for (size_t k =0; k < nslice; k++) 
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice data
      mat xslice = sortedx.rows(rownum, rownum + nh - 1);
      vec wslice = newW.subvec(rownum, rownum + nh - 1);
      
      // next slice
      rownum += nh;
      
      // center xslice
      xslice.each_row() -= mean(xslice.each_col() % wslice, 0); 
      
      // apply weight
      xslice.each_col() %= sqrt(wslice);
      
      // slice cov and M
      double w = sum(wslice);
      w = 1 - norm(wslice, "fro") / w / w;
      
      mat C = Diag - xslice.t() * xslice / w;
      M += C * C.t();
    }
    
    
  }else{
    // obs index
    size_t rownum = 0;
    
    for (size_t k =0; k < nslice; k++) 
    {
      // sample size in this slice
      // first res slices has one more obs
      size_t nh = (k < res) ? (slice_size + 1) : slice_size;
      
      // slice data
      mat xslice = sortedx.rows(rownum, rownum + nh - 1);
      
      // next slice
      rownum += nh;
      
      RLTcout << "k= " << k << std::endl;
      
      // slice cov and M
      mat C = Diag - arma::cov(xslice);
      M += C * C.t();
      
      RLTcout << "complete " << k << std::endl;
    }
  }
  
  return M;
  
}


