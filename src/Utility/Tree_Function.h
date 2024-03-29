//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Tree Arrangement Functions
//  **********************************

// my header file

# include <RcppArmadillo.h>
# include <algorithm>

using namespace Rcpp;
using namespace arma;

# include "Utility.h"
# include "Tree_Definition.h"


#ifndef RLT_TREE_FUNCTION
#define RLT_TREE_FUNCTION


// ************************//
// miscellaneous functions //
// ************************//

// categorical variables arrangement
double pack(const size_t nBits, const uvec& bits);
void unpack(const double pack, const size_t nBits, uvec& bits);
bool unpack_goright(double pack, const size_t cat);
void goright_roll_one(arma::uvec& goright_cat);

// find variable j
size_t find_j(const arma::uvec& var_id,
              size_t varj);

// sample both inbag and oobag index
void set_obstrack(arma::imat& ObsTrack,
                  const size_t nt,
                  const size_t size,
                  const bool replacement,
                  Rand& rngl);

void get_index(arma::uvec& inbag_index,
               arma::uvec& oobag_index,
               const arma::ivec& ObsTrack_nt);

// splitting an interval node
void split_id(const vec& x, double value, uvec& left_id, uvec& obs_id);
void split_id_cat(const vec& x, double value, uvec& left_id, uvec& obs_id, size_t ncat);
void split_id_comb(const mat& x, 
                   const Comb_Split_Class& OneSplit, 
                   uvec& left_id, 
                   uvec& obs_id);
  
// check cutoff points in continuous or categorical variables
void check_cont_index_sub(size_t& lowindex, size_t& highindex, const vec& x, const uvec& indices);
void check_cont_index(size_t& lowindex, size_t& highindex, const vec& x);
void move_cat_index(size_t& lowindex, 
                    size_t& highindex, 
                    std::vector<Cat_Class*>& cat_reduced, 
                    size_t true_cat, 
                    size_t nmin);

bool cat_class_compare(Cat_Class& a, Cat_Class& b);

// ***************//
// terminal nodes //
// ***************//

void Find_Terminal_Node(size_t Node, 
                        const Tree_Class& OneTree,
                        const mat& X,
                        const uvec& Ncat,
                        uvec& proxy_id,
                        const uvec& real_id,
                        uvec& TermNode);

void Find_Terminal_Node(size_t Node, 
                        const Tree_Class& OneTree,
                        const mat& X,
                        const uvec& Ncat,
                        uvec& proxy_id,
                        const uvec& real_id,
                        uvec& TermNode,
                        bool verboose);

void Find_Terminal_Node_Comb(size_t Node, 
                             const Comb_Tree_Class& OneTree,
                             const mat& X,
                             const uvec& Ncat,
                             uvec& proxy_id,
                             const uvec& real_id,
                             uvec& TermNode);

void Find_Terminal_Node_ShuffleJ(size_t Node, 
                                 const Tree_Class& OneTree,
                                 const mat& X,
                                 const uvec& Ncat,
                                 uvec& proxy_id,
                                 const uvec& real_id,
                                 uvec& TermNode,
                                 const vec& tildex,
                                 const size_t j);

void Find_Terminal_Node_Comb_ShuffleJ(size_t Node, 
                                      const Comb_Tree_Class& OneTree,
                                      const mat& X,
                                      const uvec& Ncat,
                                      uvec& proxy_id,
                                      const uvec& real_id,
                                      uvec& TermNode,
                                      const vec& tildex,
                                      const size_t j);

void Assign_Terminal_Node_Prob_RandomJ(size_t Node,
                                       const Tree_Class& OneTree,
                                       const mat& X,
                                       const uvec& Ncat,
                                       size_t id,
                                       double Multipler,
                                       vec& Prob,
                                       size_t j);

void Assign_Terminal_Node_Prob_Comb_RandomJ(size_t Node,
                                            const Comb_Tree_Class& OneTree,
                                            const mat& X,
                                            const uvec& Ncat,
                                            size_t id,
                                            double Multipler,
                                            vec& Prob,
                                            size_t j);



#endif