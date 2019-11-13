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
          if ( Y(i) > Y(j) ) // not tied i is larger
          {
              if ( Censor(j) == 1 ) // shorter is failure
              {
                  P++;
                  
                  if ( pred(i) < pred(j) ) // j has worse outcome
                      C++;
                  
                  if ( pred(i) == pred(j) )
                      C += 0.5;
                  
              }else{
                  
                  // omit  shorter is censored
              }
              
          }else if( Y(i) < Y(j) ) // not tied j is larger
          {
              
              if ( Censor(i) == 1 ) // shorter is failure
              {
                  P++;
                  
                  if ( pred(i) > pred(j) ) // i has worse oucome
                      C++;
                  
                  if ( pred(i) == pred(j) )
                      C += 0.5;              
                  
              }else{
                  
                  // omit;  shorter is censored
              }          
          }else{
              
              if ( Censor(i) + Censor(j) == 1 ) // tied but has one failure
              {
                  P++;
                  
                  if ( Censor(i) == 1 and pred(i) > pred(j) )
                      C++;
                  else if ( Censor(j) == 1 and pred(j) > pred(i) )
                      C++;
                  else
                      C+= 0.5;
                  
              }else if ( Censor(i) + Censor(j) == 2 ) // tied but has two failures
              {
                  P++;
                  
                  if ( pred(i) == pred(j) )
                      C++;
                  else
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
      
      
      
      if ( Y(i) > Y(j) ) // not tied i is larger
      {
          if ( Censor(j) == 1 ) // shorter is failure
          {
             P++;
              
             if ( pred(i) < pred(j) ) // j has worse outcome
                 C++;

             if ( pred(i) == pred(j) )
                 C += 0.5;
              
          }else{
              
              // omit  shorter is censored
          }
          
      }else if( Y(i) < Y(j) ) // not tied j is larger
      {
          
          if ( Censor(i) == 1 ) // shorter is failure
          {
              P++;
              
              if ( pred(i) > pred(j) ) // i has worse oucome
                  C++;
              
              if ( pred(i) == pred(j) )
                  C += 0.5;              

          }else{
              
              // omit;  shorter is censored
          }          
      }else{
          
          if ( Censor(i) + Censor(j) == 1 ) // tied but has one failure
          {
              P++;
              
              if ( Censor(i) == 1 and pred(i) > pred(j) )
                C++;
              else if ( Censor(j) == 1 and pred(j) > pred(i) )
                C++;
              else
                C+= 0.5;
              
          }else if ( Censor(i) + Censor(j) == 2 ) // tied but has two failures
          {
              P++;
              
              if ( pred(i) == pred(j) )
                  C++;
              else
                  C += 0.5;
          }
      }

  }}
  
  return C/P;
  
}
