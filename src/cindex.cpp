//  **********************************
//  Reinforcement Learning Trees (RLT)
//  C-index
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
double cindex_d(arma::vec& Y,
                arma::uvec& Censor,
                arma::vec& pred)
{

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

// Rcpp does not allow overloading?

double cindex_i(arma::uvec& Y,
                arma::uvec& Censor,
                arma::vec& pred)
{
  
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
