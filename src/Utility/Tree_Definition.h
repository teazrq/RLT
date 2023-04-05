//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Tree Definitions
//  **********************************

// my header file

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

#include "Utility.h"

#ifndef RLT_TREE_DEFINITION
#define RLT_TREE_DEFINITION

// *********************** //
//  Tree and forest class  //
// *********************** //

class Tree_Class{ // single split trees
public:
  arma::ivec& SplitVar;
  arma::vec& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  arma::vec& NodeWeight;
  
  Tree_Class(arma::ivec& SplitVar,
             arma::vec& SplitValue,
             arma::uvec& LeftNode,
             arma::uvec& RightNode,
             arma::vec& NodeWeight) : SplitVar(SplitVar),
                                      SplitValue(SplitValue),
                                      LeftNode(LeftNode),
                                      RightNode(RightNode),
                                      NodeWeight(NodeWeight) {}
  
  void find_next_nodes(size_t& NextLeft, size_t& NextRight)
  {
    while( SplitVar(NextLeft) != -2 ) NextLeft++;
    SplitVar(NextLeft) = -3;  
    
    NextRight = NextLeft;
    while( SplitVar(NextRight) != -2 ) NextRight++;
    
    // -2: unused, -3: reserved; Else: internal node; -1: terminal node
    SplitVar(NextRight) = -3;
  }
  
  // get tree length
  size_t get_tree_length() {
    size_t i = 0;
    while (i < SplitVar.n_elem and SplitVar(i) != -2) i++;
    return( (i < SplitVar.n_elem) ? i:SplitVar.n_elem );
  }
  
  void print() {
    
    RLTcout << "This tree has length " << get_tree_length() << std::endl;
    
  }
  
};

class Comb_Tree_Class{ // multivariate split trees
public:
  arma::imat& SplitVar;
  arma::mat& SplitLoad;
  arma::vec& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  arma::vec& NodeWeight;
  
  Comb_Tree_Class(arma::imat& SplitVar,
                   arma::mat& SplitLoad,
                   arma::vec& SplitValue,
                   arma::uvec& LeftNode,
                   arma::uvec& RightNode,
                   arma::vec& NodeWeight) : SplitVar(SplitVar),
                                            SplitLoad(SplitLoad),
                                            SplitValue(SplitValue),
                                            LeftNode(LeftNode),
                                            RightNode(RightNode),
                                            NodeWeight(NodeWeight) {}
  
  void find_next_nodes(size_t& NextLeft, size_t& NextRight)
  {
    // -2: unused, -3: reserved; Else: internal node; -1: terminal node    
    
    while( SplitVar(NextLeft, 0) != -2 ) NextLeft++;
    SplitVar(NextLeft) = -3;
    
    NextRight = NextLeft;
    
    while( SplitVar(NextRight, 0) != -2 ) NextRight++;
    SplitVar(NextRight, 0) = -3;
  }
  
  // get tree length
  size_t get_tree_length() {
    size_t i = 0;
    while (i < SplitVar.n_rows and SplitVar(i, 0) != -2) i++;
    return( (i < SplitVar.n_rows) ? i:SplitVar.n_rows );
  }
  
  void print() {
    
    RLTcout << "This tree has length " << get_tree_length() << std::endl;

  }
  
};

// **************** //
// class for splits //
// **************** //

class Split_Class{ // univariate splits
public:
  size_t var = 0;  
  double value = 0;
  double score = -1;
  
  void print(void) {
      RLTcout << "Splitting variable is " << var << " value is " << value << " score is " << score << std::endl;
  }
};

class Comb_Split_Class{ // multi-variate splits
public:
  arma::uvec& var;
  arma::vec& load;
  double value = 0;
  double score = -1;
  
  Comb_Split_Class(arma::uvec& var,
                   arma::vec& load) : 
    var(var),
    load(load) {}
  
  void print(void) {
    RLTcout << "Splitting variable is\n" << var << "load is\n" << load << "value is " << value << "; score is " << score << std::endl;
  }
};


// ************************ //
// for categorical variable //
// ************************ //

class Cat_Class{
public:
  
  virtual ~Cat_Class() = default;
  
  size_t cat = 0; // category number
  size_t count = 0; // count is used for setting nmin
  double weight = 0; // weight is used for calculation
  double score = 0; // for sorting
    
  void print() {
      RLTcout << "Category " << cat << " N = " << count << 
        " weight = " << weight << " score = " << score << std::endl;
  }
};

#endif
