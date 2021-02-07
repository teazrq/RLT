//  **********************************
//  Reinforcement Learning Trees (RLT)
//  C-index
//  **********************************

// my header file
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility/Utility.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
double PLS_test(const arma::uvec& Left_Fail, 
           const arma::uvec& Left_Risk, 
           const arma::uvec& All_Fail, 
           const arma::uvec& All_Censor, 
           const arma::uvec& All_Risk)
{
  vec etaj = conv_to< vec >::from(All_Risk);
  
  //vec tmp = (wi % All_Fail)/etaj;
  //tmp = cumsum(tmp);
  //tmp(All_Risk.n_elem-1) = 0;
  //tmp = shift(tmp, 1);
  
  vec tmpF = (All_Fail % All_Fail)/etaj;
  tmpF = cumsum(tmpF);
  tmpF(All_Risk.n_elem-1) = 0;
  tmpF = shift(tmpF, 1);
  
  vec tmpC = ((All_Censor) % All_Fail)/etaj;
  tmpC = cumsum(tmpC);
  //tmpC(All_Risk.n_elem-1) = 0;
  //tmpC = shift(tmpC, 1);
  
  //Verion without accounting for ties
  //w_eta = (etaj - 1)/(etaj%etaj);
  
  //Accouting for ties
  vec w_etaF;
  vec w_etaC;
  vec z_etaF;
  vec z_etaC;
  
  w_etaF = All_Fail % (All_Fail%etaj - All_Fail%All_Fail)/(etaj % etaj);
  w_etaF = cumsum(w_etaF);
  //Because the set of times where yj is still at risk does not include tj, shift
  w_etaF(All_Risk.n_elem-1) = 0;
  w_etaF = shift(w_etaF, 1);
  
  w_etaC = All_Fail % (All_Censor%etaj - All_Censor%All_Censor)/(etaj % etaj);
  w_etaC = cumsum(w_etaC);
  //Because the set of times where yj is still at risk includes tj, do not need to shift
  //w_etaC(All_Risk.n_elem-1) = 0;
  //w_etaC = shift(w_etaC, 1);
  
  //vec tmp2F = 1/w_etaF;
  
  z_etaF = (1/w_etaF) % (All_Fail - tmpF); //Check!!!
  z_etaF.elem( find_nonfinite(z_etaF) ).zeros();
  z_etaC = (1/w_etaC) % (0 - tmpC); //Check!!!
  z_etaC.elem( find_nonfinite(z_etaC) ).zeros();
  //z_etaC(0) = 0;
  
  // cumulative at risk count
  uvec Left_Risk_All(Left_Risk.n_elem);
  Left_Risk_All(0) = arma::accu(Left_Risk);
  
  if (Left_Risk_All(0) == 0)
    return -1;
  
  double imp = sum(w_etaF%z_etaF%Left_Fail+w_etaF%z_etaC%(Left_Risk-Left_Fail));///Left_Risk_All(0)
  double score = imp*imp;//(All_Risk(0)-Left_Risk_All(0))*imp*imp/All_Risk(0);///(Left_Risk_All(0)*Left_Risk_All(0))
  return(score);// 
}

// [[Rcpp::export()]]
double PLS_test2(const arma::uvec& Pseudo_X, 
                const arma::uvec& Y_collapse,
                const arma::uvec& Censor_collapse,
                const arma::uvec& All_Fail, 
                const arma::uvec& All_Censor, 
                const arma::uvec& All_Risk,
                const arma::uvec& Left_Risk)
{
  
  vec etaj = conv_to< vec >::from(All_Risk);
  vec z_etaj = conv_to< vec >::from(Left_Risk);
  vec tmpC1 = cumsum(z_etaj/(etaj%etaj));
  vec tmpC2 = cumsum(1/etaj);
  
  //vec tmp = (wi % All_Fail)/etaj;
  //tmp = cumsum(tmp);
  //tmp(All_Risk.n_elem-1) = 0;
  //tmp = shift(tmp, 1);
  
  vec tmpF = (All_Fail % All_Fail)/etaj;
  tmpF = cumsum(tmpF);
  tmpF(All_Risk.n_elem-1) = 0;
  tmpF = shift(tmpF, 1);
  
  vec tmpC = ((All_Censor) % All_Fail)/etaj;
  tmpC = cumsum(tmpC);
  //tmpC(All_Risk.n_elem-1) = 0;
  //tmpC = shift(tmpC, 1);
  
  //Verion without accounting for ties
  //w_eta = (etaj - 1)/(etaj%etaj);
  
  //Accouting for ties
  vec w_etaF;
  vec w_etaC;
  vec z_etaF;
  vec z_etaC;
  vec w_eta(Y_collapse.n_elem);
  vec z_eta(Y_collapse.n_elem);
  
  w_etaF = All_Fail % (All_Fail%etaj - All_Fail%All_Fail)/(etaj % etaj);
  w_etaF = cumsum(w_etaF);
  //Because the set of times where yj is still at risk does not include tj, shift
  w_etaF(All_Risk.n_elem-1) = 0;
  //w_etaF = shift(w_etaF, 1);
  
  w_etaC = All_Fail % (All_Censor%etaj - All_Censor%All_Censor)/(etaj % etaj);
  w_etaC = cumsum(w_etaC);
  //Because the set of times where yj is still at risk includes tj, do not need to shift
  //w_etaC(All_Risk.n_elem-1) = 0;
  //w_etaC = shift(w_etaC, 1);
  
  //vec tmp2F = 1/w_etaF;
  
  z_etaF = (1/w_etaF) % (All_Fail - tmpF); //Check!!!
  z_etaF.elem( find_nonfinite(z_etaF) ).zeros();
  z_etaC = (1/w_etaC) % (0 - tmpC); //Check!!!
  z_etaC.elem( find_nonfinite(z_etaC) ).zeros();
  //z_etaC(0) = 0;
  vec Cbeta(Y_collapse.n_elem);
  vec influence(Y_collapse.n_elem);
  
  for (size_t i = 0; i<Y_collapse.n_elem; i++)
  {
    Cbeta(i) = tmpC1(Y_collapse(i)) - Pseudo_X(i)*tmpC2(Y_collapse(i));
    influence(i) = Censor_collapse(i)*(Pseudo_X(i)-z_etaj(Y_collapse(i))/etaj(Y_collapse(i)))+Cbeta(i);
    
    if (Censor_collapse(i) == 1){
      z_eta(i) = z_etaF(Y_collapse(i));
      w_eta(i) = w_etaF(Y_collapse(i));
    }else{
      z_eta(i) = z_etaC(Y_collapse(i));
      w_eta(i) = w_etaF(Y_collapse(i));
    }
  }

  if (accu(Pseudo_X) == 0 or accu(Pseudo_X)==Pseudo_X.n_elem)
    return -1;
  
  //double beta = sum(w_eta%z_eta%(Pseudo_X))/sum(w_eta%Pseudo_X);
  
  //double imp = sum(w_etaF%(z_etaF)%Left_Fail+w_etaF%(z_etaC)%(Left_Risk-Left_Fail));
  double imp = sum(w_eta%z_eta%(Pseudo_X));//
  double score = imp*imp;//sum((z_eta-Pseudo_X*beta)%(z_eta-Pseudo_X*beta)%w_eta);////
  return(score);// /Left_Risk_All(0)
  //return(sum(influence%Pseudo_X)*sum(influence%Pseudo_X)/sum(Pseudo_X));
}

// [[Rcpp::export()]]
arma::vec zetaF(const arma::uvec& All_Fail, 
                const arma::uvec& All_Censor, 
                const arma::uvec& All_Risk)
{
  vec etaj = conv_to< vec >::from(All_Risk);
  
  //vec tmp = (wi % All_Fail)/etaj;
  //tmp = cumsum(tmp);
  //tmp(All_Risk.n_elem-1) = 0;
  //tmp = shift(tmp, 1);
  
  vec tmpF = (All_Fail % All_Fail)/etaj;
  tmpF = cumsum(tmpF);
  tmpF(All_Risk.n_elem-1) = 0;
  tmpF = shift(tmpF, 1);
  
  vec tmpC = ((All_Censor) % All_Fail)/etaj;
  tmpC = cumsum(tmpC);
  //tmpC(All_Risk.n_elem-1) = 0;
  //tmpC = shift(tmpC, 1);
  
  //Verion without accounting for ties
  //w_eta = (etaj - 1)/(etaj%etaj);
  
  //Accouting for ties
  vec w_etaF;
  vec w_etaC;
  vec z_etaF;
  vec z_etaC;
  
  w_etaF = All_Fail % (All_Fail%etaj - All_Fail%All_Fail)/(etaj % etaj);
  w_etaF = cumsum(w_etaF);
  //Because the set of times where yj is still at risk does not include tj, shift
  w_etaF(All_Risk.n_elem-1) = 0;
  w_etaF = shift(w_etaF, 1);
  
  w_etaC = All_Fail % (All_Censor%etaj - All_Censor%All_Censor)/(etaj % etaj);
  w_etaC = cumsum(w_etaC);
  z_etaF = (1/w_etaF) % (All_Fail - tmpF); //Check!!!
  z_etaF.elem( find_nonfinite(z_etaF) ).zeros();
  
  return(z_etaF);// 
}

// [[Rcpp::export()]]
arma::vec wetaF(const arma::uvec& All_Fail, 
                const arma::uvec& All_Censor, 
                const arma::uvec& All_Risk)
{
  vec etaj = conv_to< vec >::from(All_Risk);
  
  //vec tmp = (wi % All_Fail)/etaj;
  //tmp = cumsum(tmp);
  //tmp(All_Risk.n_elem-1) = 0;
  //tmp = shift(tmp, 1);
  
  vec tmpF = (All_Fail % All_Fail)/etaj;
  tmpF = cumsum(tmpF);
  tmpF(All_Risk.n_elem-1) = 0;
  tmpF = shift(tmpF, 1);
  
  vec tmpC = ((All_Censor) % All_Fail)/etaj;
  tmpC = cumsum(tmpC);
  //tmpC(All_Risk.n_elem-1) = 0;
  //tmpC = shift(tmpC, 1);
  
  //Verion without accounting for ties
  //w_eta = (etaj - 1)/(etaj%etaj);
  
  //Accouting for ties
  vec w_etaF;
  vec w_etaC;
  vec z_etaF;
  vec z_etaC;
  
  w_etaF = All_Fail % (All_Fail%etaj - All_Fail%All_Fail)/(etaj % etaj);
  w_etaF = cumsum(w_etaF);
  //Because the set of times where yj is still at risk does not include tj, shift
  w_etaF(All_Risk.n_elem-1) = 0;
  w_etaF = shift(w_etaF, 1);
  
  w_etaC = All_Fail % (All_Censor%etaj - All_Censor%All_Censor)/(etaj % etaj);
  w_etaC = cumsum(w_etaC);
  return(w_etaF);// 
}

// [[Rcpp::export()]]
arma::vec wetaC(const arma::uvec& All_Fail, 
               const arma::uvec& All_Censor, 
               const arma::uvec& All_Risk)
{
  vec etaj = conv_to< vec >::from(All_Risk);
  
  //vec tmp = (wi % All_Fail)/etaj;
  //tmp = cumsum(tmp);
  //tmp(All_Risk.n_elem-1) = 0;
  //tmp = shift(tmp, 1);
  
  vec tmpF = (All_Fail % All_Fail)/etaj;
  tmpF = cumsum(tmpF);
  tmpF(All_Risk.n_elem-1) = 0;
  tmpF = shift(tmpF, 1);
  
  vec tmpC = ((All_Censor) % All_Fail)/etaj;
  tmpC = cumsum(tmpC);
  //tmpC(All_Risk.n_elem-1) = 0;
  //tmpC = shift(tmpC, 1);
  
  //Verion without accounting for ties
  //w_eta = (etaj - 1)/(etaj%etaj);
  
  //Accouting for ties
  vec w_etaF;
  vec w_etaC;
  vec z_etaF;
  vec z_etaC;
  
  w_etaF = All_Fail % (All_Fail%etaj - All_Fail%All_Fail)/(etaj % etaj);
  w_etaF = cumsum(w_etaF);
  //Because the set of times where yj is still at risk does not include tj, shift
  w_etaF(All_Risk.n_elem-1) = 0;
  w_etaF = shift(w_etaF, 1);
  
  w_etaC = All_Fail % (All_Censor%etaj - All_Censor%All_Censor)/(etaj % etaj);
  w_etaC = cumsum(w_etaC);
  return(w_etaC);// 
}

// [[Rcpp::export()]]
arma::vec zetaC(const arma::uvec& All_Fail, 
                const arma::uvec& All_Censor, 
                const arma::uvec& All_Risk)
{
  vec etaj = conv_to< vec >::from(All_Risk);
  
  //vec tmp = (wi % All_Fail)/etaj;
  //tmp = cumsum(tmp);
  //tmp(All_Risk.n_elem-1) = 0;
  //tmp = shift(tmp, 1);
  
  vec tmpF = (All_Fail % All_Fail)/etaj;
  tmpF = cumsum(tmpF);
  tmpF(All_Risk.n_elem-1) = 0;
  tmpF = shift(tmpF, 1);
  
  vec tmpC = ((All_Censor) % All_Fail)/etaj;
  tmpC = cumsum(tmpC);
  //tmpC(All_Risk.n_elem-1) = 0;
  //tmpC = shift(tmpC, 1);
  
  //Verion without accounting for ties
  //w_eta = (etaj - 1)/(etaj%etaj);
  
  //Accouting for ties
  vec w_etaF;
  vec w_etaC;
  vec z_etaF;
  vec z_etaC;
  
  w_etaF = All_Fail % (All_Fail%etaj - All_Fail%All_Fail)/(etaj % etaj);
  w_etaF = cumsum(w_etaF);
  //Because the set of times where yj is still at risk does not include tj, shift
  w_etaF(All_Risk.n_elem-1) = 0;
  w_etaF = shift(w_etaF, 1);
  
  w_etaC = All_Fail % (All_Censor%etaj - All_Censor%All_Censor)/(etaj % etaj);
  w_etaC = cumsum(w_etaC);
  z_etaC = (1/w_etaC) % (0 - tmpC); //Check!!!
  z_etaC.elem( find_nonfinite(z_etaC) ).zeros();
  return(z_etaC);// 
}

// [[Rcpp::export()]]
double cindex_d(arma::vec& Y,
              arma::uvec& Censor,
              arma::vec& pred)
{

  DEBUG_Rcout << "-- calculate cindex (int Y) " << std::endl;
  size_t P = 0;
  double C = 0;
  
  for (size_t i = 0; i < Y.n_elem; i++){
      for (size_t j = 0; j < i; j ++)
      {
        if ( ( Y(i) > Y(j) and Censor(j) == 0 ) or ( Y(i) < Y(j) and Censor(i) == 0 ) )
        {
          continue;
        }
        
        if ( Y(i) == Y(j) and Censor(i) == 0 and Censor(j) == 0 )
        {
          continue;
        }
        
        P++;
        
        if ( Y(i) > Y(j) )
        {
          if ( pred(i) < pred(j) )
          {
            C++;
          }
          
          if ( pred(i) == pred(j) )
          {
            C += 0.5;
          }
          
        }else if( Y(i) < Y(j) ){
          if ( pred(j) < pred(i) )
          {
            C++;
          }
          
          if ( pred(i) == pred(j) )
          {
            C += 0.5;
          }
          
        }else{
          
          if ( Censor(i) == 1 and Censor(j) == 1 )
          {
            if ( pred(i) == pred(j) )
            {
              C++;
            }else{
              C += 0.5;
            }
              
          }else if ( ( Censor(i) == 1 and pred(i) > pred(j) ) or ( Censor(j) == 1 and pred(j) > pred(i) ) )
          {
            C++;
          }else if ( pred(i) == pred(j) )
          {
            C += 0.5;
          }
        }
      }}
  
  return C/P;
}

double cindex_i(arma::uvec& Y,
              arma::uvec& Censor,
              arma::vec& pred)
{
  
  DEBUG_Rcout << "-- calculate cindex (int Y) " << std::endl;
  size_t P = 0;
  double C = 0;
  
  for (size_t i = 0; i < Y.n_elem; i++){
    for (size_t j = 0; j < i; j ++)
    {
      if ( ( Y(i) > Y(j) and Censor(j) == 0 ) or ( Y(i) < Y(j) and Censor(i) == 0 ) )
      {
        continue;
      }
      
      if ( Y(i) == Y(j) and Censor(i) == 0 and Censor(j) == 0 )
      {
        continue;
      }
      
      P++;
      
      if ( Y(i) > Y(j) )
      {
        if ( pred(i) < pred(j) )
        {
          C++;
        }
        
        if ( pred(i) == pred(j) )
        {
          C += 0.5;
        }
        
      }else if( Y(i) < Y(j) ){
        if ( pred(j) < pred(i) )
        {
          C++;
        }
        
        if ( pred(i) == pred(j) )
        {
          C += 0.5;
        }
        
      }else{
        
        if ( Censor(i) == 1 and Censor(j) == 1 )
        {
          if ( pred(i) == pred(j) )
          {
            C++;
          }else{
            C += 0.5;
          }
          
        }else if ( ( Censor(i) == 1 and pred(i) > pred(j) ) or ( Censor(j) == 1 and pred(j) > pred(i) ) )
        {
          C++;
        }else if ( pred(i) == pred(j) )
        {
          C += 0.5;
        }
      }
    }}
  
  return C/P;
  
}




// [[Rcpp::export()]]
arma::umat ARMA_EMPTY_UMAT()
{
  arma::umat temp;
  return temp;
}

// [[Rcpp::export()]]
arma::vec ARMA_EMPTY_VEC()
{
  arma::vec temp;
  return temp;
}










