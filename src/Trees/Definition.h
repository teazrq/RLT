//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;

#define SurvWeightTH 1e-10

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
  bool pre_obstrack;
  size_t NFail;
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
    pre_obstrack  = param["pre.obs.track"];
    NFail         = param["nfail"];
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
  void initiate(size_t TreeLength)
  {
    if (TreeLength == 0) TreeLength = 1;
    
    NodeType.zeros(TreeLength);
    
    SplitVar.set_size(TreeLength);
    SplitVar.fill(datum::nan);
    
    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeAve.zeros(TreeLength);
    NodeSize.zeros(TreeLength);
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
    NodeType(span(OldLength, NewLength-1)).zeros();
    
    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength-1)).fill(datum::nan);
    
    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength-1)).zeros();
      
    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength-1)).zeros();
    
    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength-1)).zeros();
    
    NodeAve.resize(NewLength);
    NodeAve(span(OldLength, NewLength-1)).zeros();
      
    NodeSize.resize(NewLength);
    NodeSize(span(OldLength, NewLength-1)).zeros();
  }
};

// for categorical variable  

class Cat_Class{
public:
    size_t cat = 0;
    size_t count = 0; // count is used for setting nmin
    double weight = 0; // weight is used for calculation
    double score = 0; // for sorting
    
    void print() {
        Rcout << "Category is " << cat << " count is " << count << " weight is " << weight << " score is " << score << std::endl;
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
    Rcout << "Category is " << cat << " count is " << count << " weight is " << weight << " y sum is " << y << " score is " << score << std::endl;
  }
};


class Surv_Cat_Class: public Cat_Class{
public:
  arma::vec FailCount;
  arma::vec CensorCount;
  arma::vec cHaz;
  
  void initiate(size_t j, size_t NFail)
  {
	  cat = j;
	  FailCount.zeros(NFail+1);
	  CensorCount.zeros(NFail+1);
	  cHaz.zeros(NFail+1);
  }
  
  void calculate_cHaz(size_t NFail)
  {
      if (NFail == 0)
          return;
      
      double AtRisk = weight;
      double haz;
      
      AtRisk -= FailCount(0) + CensorCount(0);
      
      for (size_t k=1; k < NFail + 1; k++)
      {
          if (AtRisk > SurvWeightTH)
          {
            haz = FailCount(k) / AtRisk;
          }else{
            haz = 0;
          }
          
          cHaz[k] = cHaz[k-1] + haz;
          
          AtRisk -= FailCount(k) + CensorCount(k); 
      }
  }
  
  void set_score(size_t j)
  {
        score = cHaz(j);
  }
  
  void set_score_ccHaz()
  {
      score = sum(cHaz);
  }  
  
  void print() {
      Rcout << "Category is " << cat << " weight is " << weight << " count is " << count << " data is\n" << 
               join_rows(FailCount, CensorCount, cHaz) << std::endl;
  }  
  
};

// ************************//
// tree class for survival //
// ************************//


class Surv_Uni_Tree_Class: public Uni_Tree_Class{
public:
  arma::field<arma::vec> NodeHaz;
  
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
    NodeHaz.set_size(TreeLength);
    SplitVar.fill(P+1);
  }
  
  void trim(size_t TreeLength)
  {
    NodeType.resize(TreeLength);
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    field_vec_resize(NodeHaz, TreeLength);
    NodeSize.resize(TreeLength);
  }
  
  void extend(size_t P)
  {
    // tree is not long enough, extend
    
    size_t OldLength = NodeType.n_elem;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);
    
    NodeType.resize(NewLength);
    NodeType(span(OldLength, NewLength-1)).zeros();

    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength-1)).fill(P+1); // this should be P+1 already because of intitialization

    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength-1)).zeros();

    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength-1)).zeros();
    
    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength-1)).zeros();
    
    field_vec_resize(NodeHaz, NewLength);
    
    NodeSize.resize(NewLength);
    NodeSize(span(OldLength, NewLength-1)).zeros();
  }
};

#endif
