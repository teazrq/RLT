//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Miscellaneous Reg Functions
//  **********************************

// my header file
# include "Reg_Uni_Function.h"

using namespace Rcpp;
using namespace arma;

// Moved from Miscellaneous.cpp
void move_cat_index(size_t& lowindex, size_t& highindex, std::vector<Reg_Cat_Class>& cat_reduced, size_t true_cat, size_t nmin)
{
  lowindex = 0;
  highindex = true_cat - 2;
  
  if (true_cat == 2) //nothing we can do
    return; 
  
  size_t lowcount = cat_reduced[0].count;
  size_t highcount = cat_reduced[true_cat-1].count;
  
  // now both low and high index are not tied with the end
  if ( lowcount >= nmin and highcount >= nmin ) // everything is good
    return;
  
  if ( lowcount < nmin and highcount >= nmin ) // only need to fix lowindex
  {
    while( lowcount < nmin and lowindex <= highindex ){
      lowindex++;
      lowcount += cat_reduced[lowindex].count;
    }
    
    if ( lowindex > highindex ) lowindex = highindex;
    
    return;
    //DEBUG_Rcout << "        --- case 1 with lowindex " << lowindex << " highindex " << highindex << std::endl;
  }
  
  if ( lowcount >= nmin and highcount < nmin ) // only need to fix highindex
  {
    while( highcount < nmin and lowindex <= highindex ){
      //DEBUG_Rcout << "        --- adding " << cat_reduced[highindex].count << " count to highcount " << highcount << std::endl;
      highcount += cat_reduced[highindex].count;
      highindex--;
    }
    
    if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex; // sometimes highindex will be negative and turned into very large number 
    
    //DEBUG_Rcout << "        --- case 2 with lowindex " << lowindex << " highindex " << highindex << std::endl;    
    
    return;
  }
  
  if ( lowcount < nmin and highcount < nmin ) // if both need to be fixed, start with one randomly
  {
    if ( TRUE ) //************ Later ********************//
    { // fix lowindex first
      while( lowcount < nmin and lowindex <= highindex ){
        lowindex++;
        lowcount += cat_reduced[lowindex].count;
      }
      
      if (lowindex > highindex ) lowindex = highindex;
      
      while( highcount < nmin and lowindex <= highindex ){
        highcount += cat_reduced[highindex].count;
        highindex--;
      }
      
      if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex;
      
      //DEBUG_Rcout << "        --- case 3 with lowindex " << lowindex << " highindex " << highindex << std::endl;
      return;
      
    }else{ // fix highindex first
      while( highcount < nmin and lowindex <= highindex ){
        highcount += cat_reduced[highindex].count;
        highindex--;
      }
      
      if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex;
      
      while( lowcount < nmin and lowindex <= highindex ){
        lowindex++;
        lowcount += cat_reduced[lowindex].count;
      }
      
      if (lowindex > highindex) lowindex = highindex;
      
      //DEBUG_Rcout << "        --- case 4 with lowindex " << lowindex << " highindex " << highindex << std::endl;
      return;
    }
  }
}

//Moved from Miscellaneous.cpp
double record_cat_split(std::vector<Reg_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat)
{
  uvec goright(ncat + 1, fill::zeros); // the first element (category) of goright will always be set to 0 --- go left, but this category does not exist.
  
  for (size_t i = 0; i <= best_cat; i++)
    goright[cat_reduced[i].cat] = 0;
  
  for (size_t i = best_cat + 1; i < true_cat; i++)
    goright[cat_reduced[i].cat] = 1;
  
  for (size_t i = true_cat + 1; i < ncat + 1; i++)
    goright[cat_reduced[i].cat] = 0; // for empty category, assign randomly
  
  return pack(ncat + 1, goright);
}
