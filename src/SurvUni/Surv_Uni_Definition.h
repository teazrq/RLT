//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header files
# include "../Utility/Tree_Definition.h"
# include "../Utility/Utility.h"
# include "../Utility/Tree_Function.h"

using namespace Rcpp;
using namespace arma;

#ifndef RLT_SURV_UNI_DEFINITION
#define RLT_SURV_UNI_DEFINITION

// ************ //
//  data class  //
// ************ //

class RLT_SURV_DATA{
public:
  arma::mat& X;
  arma::uvec& Y;
  arma::uvec& Censor;
  arma::uvec& Ncat;
  arma::size_t& NFail;
  arma::vec& obsweight;
  arma::vec& varweight;
  
  RLT_SURV_DATA(arma::mat& X, 
               arma::uvec& Y,
               arma::uvec& Censor,
               arma::uvec& Ncat,
               arma::size_t& NFail,
               arma::vec& obsweight,
               arma::vec& varweight) : X(X), 
               Y(Y), 
               Censor(Censor),
               Ncat(Ncat), 
               NFail(NFail),
               obsweight(obsweight), 
               varweight(varweight) {}
};

// forest class survival 

class Surv_Uni_Forest_Class{
public:
  arma::field<arma::ivec>& SplitVarList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeWeightList;
  arma::field<arma::field<arma::vec>>& NodeHazList;
  
  Surv_Uni_Forest_Class(arma::field<arma::ivec>& SplitVarList,
                       arma::field<arma::vec>& SplitValueList,
                       arma::field<arma::uvec>& LeftNodeList,
                       arma::field<arma::uvec>& RightNodeList,
                       arma::field<arma::vec>& NodeWeightList,
                       arma::field<arma::field<arma::vec>>& NodeHazList) : SplitVarList(SplitVarList),
                                                                           SplitValueList(SplitValueList),
                                                                           LeftNodeList(LeftNodeList),
                                                                           RightNodeList(RightNodeList),
                                                                           NodeWeightList(NodeWeightList),
                                                                           NodeHazList(NodeHazList) {}
};

// tree class survival 

class Surv_Uni_Tree_Class : public Tree_Class{
public:
  arma::field<arma::vec>& NodeHaz;
  
  Surv_Uni_Tree_Class(arma::ivec& SplitVar,
                      arma::vec& SplitValue,
                      arma::uvec& LeftNode,
                      arma::uvec& RightNode,
                      arma::vec& NodeWeight,
                      arma::field<arma::vec>& NodeHaz) : Tree_Class(SplitVar,
                                                                    SplitValue,
                                                                    LeftNode,
                                                                    RightNode,
                                                                    NodeWeight),
                                                         NodeHaz(NodeHaz) {}

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
    NodeHaz.set_size(TreeLength);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeWeight.resize(TreeLength);
    field_vec_resize(NodeHaz, TreeLength);
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
    
    field_vec_resize(NodeHaz, NewLength);
  }
};

class Surv_Cat_Class: public Cat_Class{
public:
  arma::uvec FailCount;
  arma::uvec RiskCount;
  size_t nfail; 
  
  void initiate(size_t j, size_t NFail)
  {
    cat = j;
    nfail = 0;
    FailCount.zeros(NFail+1);
    RiskCount.zeros(NFail+1);
  }
  
  void print(void) {
    RLTcout << "Category is " << cat << " count is " << count << " nfail is " << nfail << std::endl;
  }
  
};

#endif
