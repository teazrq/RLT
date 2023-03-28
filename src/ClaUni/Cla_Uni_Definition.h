//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header files
# include "../Utility/Tree_Definition.h"
# include "../Utility/Utility.h"
# include "../Utility/Tree_Function.h"

using namespace Rcpp;
using namespace arma;

#ifndef RLT_CLA_UNI_DEFINITION
#define RLT_CLA_UNI_DEFINITION

// ************ //
//  data class  //
// ************ //

class RLT_CLA_DATA{
public:
  arma::mat& X;
  arma::uvec& Y;
  arma::uvec& Ncat;
  size_t nclass;
  arma::vec& obsweight;
  arma::vec& varweight;
  
  RLT_CLA_DATA(arma::mat& X, 
               arma::uvec& Y,
               arma::uvec& Ncat,
               size_t nclass,
               arma::vec& obsweight,
               arma::vec& varweight) : X(X), 
                                       Y(Y), 
                                       Ncat(Ncat), 
                                       nclass(nclass),
                                       obsweight(obsweight), 
                                       varweight(varweight) {}
};

// forest class classification 

class Cla_Uni_Forest_Class{
public:
  arma::field<arma::ivec>& SplitVarList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeWeightList;
  arma::field<arma::mat>& NodeProbList;
  
  Cla_Uni_Forest_Class(arma::field<arma::ivec>& SplitVarList,
                       arma::field<arma::vec>& SplitValueList,
                       arma::field<arma::uvec>& LeftNodeList,
                       arma::field<arma::uvec>& RightNodeList,
                       arma::field<arma::vec>& NodeWeightList,
                       arma::field<arma::mat>& NodeProbList) : SplitVarList(SplitVarList), 
                                                               SplitValueList(SplitValueList),
                                                               LeftNodeList(LeftNodeList),
                                                               RightNodeList(RightNodeList),
                                                               NodeWeightList(NodeWeightList),
                                                               NodeProbList(NodeProbList) {}
};

class Cla_Uni_Tree_Class : public Tree_Class{
public:
  arma::mat& NodeProb;

  Cla_Uni_Tree_Class(arma::ivec& SplitVar,
                     arma::vec& SplitValue,
                     arma::uvec& LeftNode,
                     arma::uvec& RightNode,
                     arma::vec& NodeWeight,
                     arma::mat& NodeProb) : Tree_Class(SplitVar,
                                                      SplitValue,
                                                      LeftNode,
                                                      RightNode,
                                                      NodeWeight),
                                            NodeProb(NodeProb) {}

  // initiate tree
  void initiate(size_t TreeLength, size_t nclass)
  {
    if (TreeLength == 0) TreeLength = 1;

    SplitVar.set_size(TreeLength);
    SplitVar.fill(-2);
    SplitVar(0) = -3;

    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeWeight.zeros(TreeLength);
    NodeProb.zeros(TreeLength, nclass);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeWeight.resize(TreeLength);
    NodeProb.resize(TreeLength, NodeProb.n_cols);
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
    
    NodeProb.resize(NewLength, NodeProb.n_cols);
    NodeProb.rows(OldLength, NewLength-1).zeros();
  }
};


class Cla_Cat_Class: public Cat_Class{
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
                    std::vector<Cla_Cat_Class>& cat_reduced, 
                    size_t true_cat, 
                    size_t nmin);

//Record category
double record_cat_split(std::vector<Cla_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat);
#endif
