//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;

#ifndef RLT_DEFINITION
#define RLT_DEFINITION

class PARAM_GLOBAL{
public:
  size_t N;
  size_t P;
  size_t ntrees;
  size_t mtry;
  size_t nmin;
  double alpha;
  int split_gen;
  int split_rule;
  int nsplit;
  bool replacement;
  double resample_prob;
  bool useobsweight;
  bool usevarweight;
  int importance;
  bool reinforcement;
  bool kernel_ready;
  int seed;
  
  PARAM_GLOBAL(List& param){
    N             = param["n"];
    P             = param["p"];
    ntrees        = param["ntrees"];
    mtry          = param["mtry"];
    nmin          = param["nmin"];
    alpha         = param["alpha"];
    split_gen     = param["split.gen"];
    split_rule    = param["split.rule"];
    nsplit        = param["nsplit"];
    replacement   = param["replacement"];
    kernel_ready  = param["kernel.ready"];
    resample_prob = param["resample.prob"];
    importance    = param["importance"];
    reinforcement = param["reinforcement"];
    useobsweight  = param["use.obs.w"];
    usevarweight  = param["use.var.w"];
    seed          = param["seed"];
  }
};

class PARAM_RLT{
public:
  bool reinforcement = 0;
};

// ****************//
// field functions //
// ****************//

void field_vec_resize(arma::field<arma::vec>& A, size_t size);

// **************************//
// class for tree and splits //
// **************************//

class Base_Tree_Class{
public:
  arma::uvec NodeType;
  arma::uvec LeftNode;
  arma::uvec RightNode;
  arma::vec NodeSize;
  
  // get tree length
  size_t get_tree_length() {
    size_t i = 0;
    while (i < NodeType.n_elem and NodeType(i) != 0) i++;
    return( (i < NodeType.n_elem) ? i:NodeType.n_elem );
  }
  
  // find the next left and right nodes 
  void find_next_nodes(size_t& NextLeft, size_t& NextRight)
  {
    while( NodeType(NextLeft) ) NextLeft++;
    NodeType(NextLeft) = 1;  
    
    NextRight = NextLeft;
    while( NodeType(NextRight) ) NextRight++;
    
    // 0: unused, 1: reserved; 2: internal node; 3: terminal node
    NodeType(NextRight) = 1;
  }
};

class Uni_Tree_Class: public Base_Tree_Class{ // univariate split trees
public:
  arma::uvec SplitVar;
  arma::vec SplitValue;
  
  void readin(arma::uvec& NodeType_R, arma::uvec& SplitVar_R, arma::vec& SplitValue_R,
              arma::uvec& LeftNode_R, arma::uvec& RightNode_R)
  {
    NodeType = uvec(NodeType_R.begin(), NodeType_R.size(), false, true);
    SplitVar = uvec(SplitVar_R.begin(), SplitVar_R.size(), false, true);
    SplitValue = vec(SplitValue_R.begin(), SplitValue_R.size(), false, true);
    LeftNode = uvec(LeftNode_R.begin(), LeftNode_R.size(), false, true);
    RightNode = uvec(RightNode_R.begin(), RightNode_R.size(), false, true);
  }
};

class Multi_Tree_Class: public Base_Tree_Class{ // multivariate split trees
public:
  arma::field<arma::uvec> SplitVar;
  arma::field<arma::vec> SplitLoad;
  arma::vec SplitValue;
};

class Uni_Split_Class{ // univariate splits
public:
  size_t var = 0;  
  double value = 0;
  double score = -1;
  
  void print(void) {
    Rcout << "Splitting varible is " << var << " value is " << value << " score is " << score << std::endl;
  }
};

class Base_Cat_Class{ // class variable reduced data
public:
  size_t cat = 0;
  size_t count = 0;
  double weight = 0;
};


// *********************//
// class for regression //
// *********************//

class Reg_Uni_Tree_Class: public Uni_Tree_Class{
public:
  arma::vec NodeAve;
  
  // readin from R 
  void readin(arma::uvec& NodeType_R, arma::uvec& SplitVar_R, arma::vec& SplitValue_R,
              arma::uvec& LeftNode_R, arma::uvec& RightNode_R, arma::vec& NodeAve_R, 
              arma::vec& NodeSize_R)
  {
    NodeType = uvec(NodeType_R.begin(), NodeType_R.size(), false, true);
    SplitVar = uvec(SplitVar_R.begin(), SplitVar_R.size(), false, true);
    SplitValue = vec(SplitValue_R.begin(), SplitValue_R.size(), false, true);
    LeftNode = uvec(LeftNode_R.begin(), LeftNode_R.size(), false, true);
    RightNode = uvec(RightNode_R.begin(), RightNode_R.size(), false, true);
    NodeAve = vec(NodeAve_R.begin(), NodeAve_R.size(), false, true);
    NodeSize = vec(NodeSize_R.begin(), NodeSize_R.size(), false, true);
  }
  
  // initiate tree
  void initiate(size_t TreeLength, size_t P)
  {
    if (TreeLength == 0) TreeLength = 1;
    if (P == 0) P = 1;
    
    NodeType.zeros(TreeLength);
    SplitVar.zeros(TreeLength);
    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeAve.zeros(TreeLength);
    NodeSize.zeros(TreeLength);
    
    SplitVar.fill(P+1);
  }

  void trim(size_t TreeLength)
  {
    NodeType.resize(TreeLength);
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeAve.resize(TreeLength);
    NodeSize.resize(TreeLength);
  }

  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = NodeType.n_elem;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);
    
    NodeType.resize(NewLength);
    NodeType(span(OldLength, NewLength-1)) = 0;
    
    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength-1)) = SplitVar[0]; // this should be P+1 already because intitialization
    
    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength-1)) = 0;
      
    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength-1)) = 0;
    
    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength-1)) = 0;
    
    NodeAve.resize(NewLength);
    NodeAve(span(OldLength, NewLength-1)) = 0;
      
    NodeSize.resize(NewLength); // need to remove later 
    NodeSize(span(OldLength, NewLength-1)) = 0;
  }
};

// for classification 

class Reg_Cat_Class: public Base_Cat_Class{
public:
  double y = 0;
  
  void print(void) {
    Rcout << "Category is " << cat << " count is " << count << " weight is " << weight << " y sum is " << y << std::endl;
  }
};


// *******************//
// class for survival //
// *******************//


class Surv_Uni_Tree_Class: public Uni_Tree_Class{
public:
  arma::field<arma::vec> NodeSurv;
  
  // initiate tree
  void initiate(size_t TreeLength, size_t P)
  {
    if (TreeLength == 0) TreeLength = 1;
    if (P == 0) P = 1;
    
    NodeType.zeros(TreeLength);
    SplitVar.zeros(TreeLength);
    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeSize.zeros(TreeLength);
    NodeSurv.set_size(TreeLength);
    SplitVar.fill(P+1);
  }
  
  void trim(size_t TreeLength)
  {
    NodeType.resize(TreeLength);
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    field_vec_resize(NodeSurv, TreeLength);
    NodeSize.resize(TreeLength);
  }
  
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = NodeType.n_elem;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);
    
    NodeType.resize(NewLength);
    NodeType(span(OldLength, NewLength-1)) = 0;
    
    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength-1)) = SplitVar[0]; // this should be P+1 already because of intitialization
    
    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength-1)) = 0;
    
    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength-1)) = 0;
    
    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength-1)) = 0;
    
    field_vec_resize(NodeSurv, NewLength);
    
    NodeSize.resize(NewLength);
    NodeSize(span(OldLength, NewLength-1)) = 0;
  }
};

#endif
