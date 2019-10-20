//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Utility Functions and Definitions
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"

using namespace Rcpp;
using namespace arma;

// ******* //
//  Debug  //
// ******* //

// this debug function will output results to a .txt file
void printLog(const char*, const char*, const int, const double);

#ifdef RLT_DEBUG
#define DEBUGPRINT(mode, x, n1, n2) printLog(mode, x, n1, n2)
#else
#define DEBUGPRINT(mode, x, n1, n2)
#endif

// this debug function will output to R
#ifdef RLT_DEBUG
#define DEBUG_Rcout Rcout
#else
#define DEBUG_Rcout 0 && Rcout
#endif




#ifndef RLT_UTILITY
#define RLT_UTILITY

// ****************//
// Check functions //
// ****************//

int checkCores(int, int);

// *************//
// Calculations //
// *************//

template <class T> const T& max (const T& a, const T& b);
template <class T> const T& min (const T& a, const T& b);

// ************************//
// Random Number Generator //
// ************************//

int intRand(const int & min, const int & max);

#endif