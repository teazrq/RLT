//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header files
# include "../Utility/Tree_Definition.h"
# include "../Utility/Utility.h"
# include "../Utility/Tree_Function.h"

using namespace Rcpp;
using namespace arma;

#ifndef RLT_REG_UNI_DEFINITION
#define RLT_REG_UNI_DEFINITION

// ************ //
//  data class  //
// ************ //

class RLT_REG_DATA{
public:
  arma::mat& X;
  arma::vec& Y;
  arma::uvec& Ncat;
  arma::vec& obsweight;
  arma::vec& varweight;
  
  RLT_REG_DATA(arma::mat& X, 
               arma::vec& Y,
               arma::uvec& Ncat,
               arma::vec& obsweight,
               arma::vec& varweight) : X(X), 
               Y(Y), 
               Ncat(Ncat), 
               obsweight(obsweight), 
               varweight(varweight) {}
};

// forest class regression 

class Reg_Uni_Forest_Class{
public:
  arma::field<arma::ivec>& SplitVarList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeWeightList;
  arma::field<arma::vec>& NodeAveList;
  
  Reg_Uni_Forest_Class(arma::field<arma::ivec>& SplitVarList,
                       arma::field<arma::vec>& SplitValueList,
                       arma::field<arma::uvec>& LeftNodeList,
                       arma::field<arma::uvec>& RightNodeList,
                       arma::field<arma::vec>& NodeWeightList,
                       arma::field<arma::vec>& NodeAveList) : SplitVarList(SplitVarList), 
                                                              SplitValueList(SplitValueList),
                                                              LeftNodeList(LeftNodeList),
                                                              RightNodeList(RightNodeList),
                                                              NodeWeightList(NodeWeightList),
                                                              NodeAveList(NodeAveList) {}
};

class Reg_Uni_Tree_Class : public Tree_Class{
public:
  arma::vec& NodeAve;

  Reg_Uni_Tree_Class(arma::ivec& SplitVar,
                     arma::vec& SplitValue,
                     arma::uvec& LeftNode,
                     arma::uvec& RightNode,
                     arma::vec& NodeWeight,
                     arma::vec& NodeAve) : Tree_Class(SplitVar,
                                                      SplitValue,
                                                      LeftNode,
                                                      RightNode,
                                                      NodeWeight),
                                                      NodeAve(NodeAve) {}

  // initiate tree
  void initiate(size_t TreeLength)
  {
    if (TreeLength == 0) TreeLength = 1;

    SplitVar.set_size(TreeLength);
    SplitVar.fill(-2);
    SplitVar(0) = -3;

    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeWeight.zeros(TreeLength);
    NodeAve.zeros(TreeLength);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeWeight.resize(TreeLength);
    NodeAve.resize(TreeLength);
  }

  // extend tree
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = SplitVar.n_elem;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);

    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength-1)).fill(-2);

    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength-1)).zeros();

    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength-1)).zeros();

    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength-1)).zeros();

    NodeWeight.resize(NewLength);
    NodeWeight(span(OldLength, NewLength-1)).zeros();
    
    NodeAve.resize(NewLength);
    NodeAve(span(OldLength, NewLength-1)).zeros();
  }
};


class Reg_Uni_Comb_Forest_Class{
public:
  arma::field<arma::imat>& SplitVarList;
  arma::field<arma::mat>& SplitLoadList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeWeightList;
  arma::field<arma::vec>& NodeAveList;
  
  Reg_Uni_Comb_Forest_Class(arma::field<arma::imat>& SplitVarList,
                         arma::field<arma::mat>& SplitLoadList,
                         arma::field<arma::vec>& SplitValueList,
                         arma::field<arma::uvec>& LeftNodeList,
                         arma::field<arma::uvec>& RightNodeList,
                         arma::field<arma::vec>& NodeWeightList,
                         arma::field<arma::vec>& NodeAveList) : SplitVarList(SplitVarList), 
                                                                SplitLoadList(SplitLoadList), 
                                                                SplitValueList(SplitValueList),
                                                                LeftNodeList(LeftNodeList),
                                                                RightNodeList(RightNodeList),
                                                                NodeWeightList(NodeWeightList),
                                                                NodeAveList(NodeAveList) {}
};

class Reg_Uni_Comb_Tree_Class : public Comb_Tree_Class{
public:
  arma::vec& NodeAve;
  
  Reg_Uni_Comb_Tree_Class(arma::imat& SplitVar,
                          arma::mat& SplitLoad,
                          arma::vec& SplitValue,
                          arma::uvec& LeftNode,
                          arma::uvec& RightNode,
                          arma::vec& NodeWeight,
                          arma::vec& NodeAve) : Comb_Tree_Class(SplitVar,
                                                                SplitLoad,
                                                                SplitValue,
                                                                LeftNode,
                                                                RightNode,
                                                                NodeWeight),
                                                NodeAve(NodeAve) {}
  
  // initiate tree
  void initiate(size_t TreeLength, size_t linear_comb)
  {
    if (TreeLength == 0) TreeLength = 1;
    if (linear_comb <= 1) stop("Linear Combination is not needed, something wrong...");
      
    SplitVar.zeros(TreeLength, linear_comb);
    SplitVar.col(0).fill(-2);
    
    SplitLoad.zeros(TreeLength, linear_comb);
    
    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeWeight.zeros(TreeLength);
    NodeAve.zeros(TreeLength);
  }
  
  // trim tree
  void trim(size_t TreeLength)
  {
    SplitVar.resize(TreeLength, SplitVar.n_cols);
    SplitLoad.resize(TreeLength, SplitVar.n_cols);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeWeight.resize(TreeLength);
    NodeAve.resize(TreeLength);
  }
  
  // extend tree
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = SplitVar.n_rows;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);
    
    SplitVar.resize(NewLength, SplitVar.n_cols);
    SplitVar.rows(OldLength, NewLength-1).zeros();
    SplitVar.submat(OldLength, 0, NewLength-1, 0).fill(-2);
    
    SplitLoad.resize(NewLength, SplitLoad.n_cols);
    SplitLoad.rows(OldLength, NewLength-1).zeros();
    
    SplitValue.resize(NewLength);
    SplitValue.subvec(OldLength, NewLength-1).zeros();
    
    LeftNode.resize(NewLength);
    LeftNode.subvec(OldLength, NewLength-1).zeros();
    
    RightNode.resize(NewLength);
    RightNode.subvec(OldLength, NewLength-1).zeros();

    NodeWeight.resize(NewLength);
    NodeWeight.subvec(OldLength, NewLength-1).zeros();
    
    NodeAve.resize(NewLength);
    NodeAve.subvec(OldLength, NewLength-1).zeros();
  }
};


class Reg_Cat_Class: public Cat_Class{
public:
  double y = 0;

  void calculate_score()
  {
    if (weight > 0)
      score = y / weight;
  }

  void print(void) {
    RLTcout << "Category is " << cat << " count is " << count << " weight is " << weight << " y sum is " << y << " score is " << score << std::endl;
  }
};

//Move categorical index
void move_cat_index(size_t& lowindex, 
                    size_t& highindex, 
                    std::vector<Reg_Cat_Class>& cat_reduced, 
                    size_t true_cat, 
                    size_t nmin);

//Record category
double record_cat_split(std::vector<Reg_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat);
#endif
